from cgi import print_directory
import os
import gc
import argparse
import json
import random
import math
import random
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
import torch.distributed as dist
import torch.multiprocessing as mp
from performer_pytorch import PerformerLM
import scanpy as sc
import anndata as ad
from utils import *
from tqdm import tqdm
from datetime import datetime
from time import time
from collections import OrderedDict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_addr", type=str, default="127.0.0.1", help='Master addr for dist training.')
    parser.add_argument("--master_port", type=str, default="8500", help='Master port for dist training.')
    parser.add_argument("--world_size", type=int, default=2, help='Number of GPUs.')
    parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
    parser.add_argument("--gene_num", type=int, default=None, help='Number of genes.') # 16906, if not supplied, will take the number of genes in the supplied training data
    parser.add_argument("--epochs", type=int, default=100, help='Number of epochs.')
    parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
    parser.add_argument("--batch_size", type=int, default=3, help='Batch size.')
    parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
    parser.add_argument("--grad_acc", type=int, default=60, help='Number of gradient accumulation.')
    parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
    parser.add_argument("--mask_prob", type=float, default=0.15, help='Probability of masking.')
    parser.add_argument("--replace_prob", type=float, default=0.9, help='Probability of replacing with [MASK] token for masking.')
    parser.add_argument("--pos_embed_g2v", action='store_true', help='Using Gene2vec encoding or not (default no unless this arg is supplied).')
    parser.add_argument("--sin_emb_wavelength", type=float, default = None, help='Wavelength of sinusoidal expression encodings. Defaults to bin_num.')
    parser.add_argument("--small_geneset", action='store_true', help='Train a smaller model. Currently implemented as including genes present in at least 5% of cells.')
    parser.add_argument("--g2v_file", type=str, default='/data/rna_rep_learning/scBERT/gene2vec_16906.npy', help='File containing Gene2vec embeddings')
    parser.add_argument("--data_path", type=str, default='/data/rna_rep_learning/scBERT/panglao_human.h5ad', help='Path of data for pretraining.')
    parser.add_argument("--ckpt_dir", type=str, default='./ckpts/', help='Directory of checkpoint to save.')
    parser.add_argument("--model_name", type=str, default='panglao_pretrain', help='Model name used for saving model.')
    parser.add_argument("--pretrained_ckpt", type=str, default=None, help='Pretrained checkpoint path.')
    parser.add_argument("--pred_continuous", action="store_true", help='If this arg is provided, embed continuous expression values and predict continuous expression values during masking, instead of bucketed.')
    parser.add_argument("--debug", action="store_true", help="Debug setting: saves to new dir.")
    args = parser.parse_args()

    model_name = args.model_name

    # Control sources of randomness
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # If continuing training from checkpoint
    if args.pretrained_ckpt and not args.debug:
        ckpt_dir = os.path.dirname(args.pretrained_ckpt)
    else:
        timestamp = datetime.now().strftime("%Y-%b-%d-%H:%M:%S")
        ckpt_dir = os.path.join(args.ckpt_dir, model_name, timestamp)

    # Create checkpoint dir if doesn't exist
    # NOTE: Done before distributing to avoid process collision
    if not (os.path.exists(ckpt_dir)):
        os.makedirs(ckpt_dir)
    
    print("Checkpoint dir: ", ckpt_dir)

    mp.spawn(
        distributed_pretrain,
        args=(args, ckpt_dir, model_name),
        nprocs=args.world_size,
        join=True,
    )


def distributed_pretrain(rank, args, ckpt_dir, model_name):

    SEED = args.seed
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    GRADIENT_ACCUMULATION = args.grad_acc
    LEARNING_RATE = args.learning_rate
    VALIDATE_EVERY = args.valid_every
    CLASS = args.bin_num + 2
    POS_EMBED_USING = args.pos_embed_g2v
    if args.sin_emb_wavelength:
        SIN_EMB_WAVELENGTH = args.sin_emb_wavelength
    else:
        SIN_EMB_WAVELENGTH = args.bin_num
    MASK_PROB = args.mask_prob
    REPLACE_PROB = args.replace_prob
    PRED_CONTINUOUS = args.pred_continuous
    RANDOM_TOKEN_PROB = 0.
    MASK_TOKEN_ID = CLASS - 1
    PAD_TOKEN_ID = CLASS - 1
    MASK_IGNORE_TOKEN_IDS = [0]

    is_master = rank == 0
    master_addr = args.master_addr
    master_port = args.master_port
    world_size = args.world_size

    ### HELPER FUNCTIONS AND DATASET CLASS FROM ORIGINAL CODE ###

    # get the random prob matrix and True means smaller than prob threshold
    def prob_mask_like(t, prob):
        return torch.zeros_like(t).float().uniform_(0, 1) < prob

    # get the mask matrix which cannot be masked
    def mask_with_tokens(t, token_ids):
        init_no_mask = torch.full_like(t, False, dtype=torch.bool)
        mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
        return mask

    def get_mask_subset_with_prob(mask, prob):
        batch, seq_len, device = *mask.shape, mask.device
        max_masked = math.ceil(prob * seq_len)      # num of mask of a single sequence in average
        num_tokens = mask.sum(dim=-1, keepdim=True)     # num of pure tokens of each sequence except special tokens
        mask_excess = torch.cat((torch.zeros(0), torch.arange(mask.size(-1)).repeat(mask.size(0)))).reshape(mask.size(0),mask.size(-1)).to(device)
        mask_excess = (mask_excess >= (num_tokens * prob).ceil())        # only 15% of pure tokens can be masked
        mask_excess = mask_excess[:, :max_masked]       # get difference between 15% of pure tokens and 15% of all tokens
        rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)     # rand (0-1) as prob, special token use -1e9
        _, sampled_indices = rand.topk(max_masked, dim=-1)      # get index of topk prob to mask
        sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)        # delete difference of mask not pure
        new_mask = torch.zeros((batch, seq_len + 1), device=device)     # get (batch, seq_len) shape zero matrix
        new_mask.scatter_(-1, sampled_indices, 1)       # set masks in zero matrix as 1
        return new_mask[:, 1:].bool()       # the final mask, True is mask

    def data_mask(
        data,
        mask_prob = MASK_PROB,
        replace_prob = REPLACE_PROB,
        num_tokens = None,
        random_token_prob = RANDOM_TOKEN_PROB,
        mask_token_id = MASK_TOKEN_ID,
        pad_token_id = PAD_TOKEN_ID,
        mask_ignore_token_ids = MASK_IGNORE_TOKEN_IDS
    ):
        mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])
        # do not mask [pad] tokens, or any other tokens in the tokens designated to be excluded ([cls], [sep])
        # also do not include these special tokens in the tokens chosen at random
        no_mask = mask_with_tokens(data, mask_ignore_token_ids)   # ignore_token as True, will not be masked later
        mask = get_mask_subset_with_prob(~no_mask, mask_prob)      # get the True/False mask matrix
        # get mask indices
        ## mask_indices = torch.nonzero(mask, as_tuple=True)   # get the index of mask(nonzero value of mask matrix)
        # mask input with mask tokens with probability of `replace_prob` (keep tokens the same with probability 1 - replace_prob)
        masked_input = data.clone().detach()
        # if random token probability > 0 for mlm
        if random_token_prob > 0:
            assert num_tokens is not None, 'num_tokens keyword must be supplied when instantiating MLM if using random token replacement'
            random_token_prob = prob_mask_like(data, random_token_prob)       # get the mask matrix of random token replace
            random_tokens = torch.randint(0, num_tokens, data.shape, device=data.device)     # generate random token matrix with the same shape as input
            random_no_mask = mask_with_tokens(random_tokens, mask_ignore_token_ids)        # not masked matrix for the random token matrix
            random_token_prob &= ~random_no_mask        # get the pure mask matrix of random token replace
            random_indices = torch.nonzero(random_token_prob, as_tuple=True)        # index of random token replace
            masked_data[random_indices] = random_tokens[random_indices]        # replace some tokens by random token
        # [mask] input
        replace_prob = prob_mask_like(data, replace_prob)     # get the mask matrix of token being masked
        masked_input = masked_input.masked_fill(mask * replace_prob, mask_token_id)        # get the data has been masked by mask_token
        # mask out any tokens to padding tokens that were not originally going to be masked
        labels = data.masked_fill(~mask, pad_token_id)        # the label of masked tokens; will have "pad_token_id" everywhere that was not masked (eg. of pad_token_id having overloaded uses...)
        return masked_input, labels

    def MSEloss(preds, target, reduction = 'mean', ignore_index = MASK_TOKEN_ID):
        """
        Created our own function to allow for an "ignore_index" argument
        """
        if not (target.size() == preds.size()):
            print(
                "Using a target size ({}) that is different to the input size ({}). "
                #"This will likely lead to incorrect results due to broadcasting. "
                "Please ensure they have the same size.".format(target.size(), preds.size())
            )
        if reduction != "mean":
            print("WARNING: mean MSEloss is automatically calculated, even though you specified a different reduction")
        #expanded_preds, expanded_target = torch.broadcast_tensors(preds, target)
        diff = (preds-target)*(target!=ignore_index)      #dont count loss from values that were not masked
        return torch.mean(diff**2)

    class SCDataset(Dataset):
        def __init__(self, data, use_continuous=False):
            super().__init__()
            self.data = data
            self.use_continuous = use_continuous

        def __getitem__(self, index):
            rand_start = random.randint(0, self.data.shape[0]-1)
            full_seq = self.data[rand_start].toarray()[0]
            full_seq[full_seq > (CLASS - 2)] = CLASS - 2
            full_seq = torch.from_numpy(full_seq)
            if(not self.use_continuous):
                full_seq = full_seq.long() #long() converts to int64
            full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
            return full_seq

        def __len__(self):
            return self.data.shape[0]

    cur_time = time()
    setup_process(rank, master_addr, master_port, world_size)
    device = torch.device("cuda:{}".format(rank))

    print("Set up distributed processes...")

    data = sc.read_h5ad(args.data_path)
    if args.debug:
        debug_seq_len = 5000
        data = data[:50,:debug_seq_len]
        GRADIENT_ACCUMULATION = 1
    elif args.small_geneset:
        sc.pp.filter_genes(data, min_cells=0.05*len(data))
        print("Filtered data to include {} genes present in at least 5% of cells".format(data.shape[1]))
    data = data.X
    if args.gene_num is not None:
        SEQ_LEN = args.gene_num + 1
    else:
        SEQ_LEN = data.shape[1] + 1 # num_genes + 1
    
    data_train, data_val = train_test_split(data, test_size=0.05, random_state=SEED)

    train_dataset = SCDataset(data_train, PRED_CONTINUOUS)
    val_dataset = SCDataset(data_val, PRED_CONTINUOUS)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = SequentialDistributedSampler(val_dataset, batch_size=BATCH_SIZE, world_size=world_size)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

    print("Loaded data...")

    # model
    model = PerformerLM(
        num_tokens = CLASS,
        dim = 200,
        depth = 6,
        max_seq_len = SEQ_LEN,
        heads = 10,
        local_attn_heads = 0,
        g2v_position_emb = POS_EMBED_USING,
        g2v_file = args.g2v_file,
        pred_continuous = PRED_CONTINUOUS,
        sin_emb_wavelength = SIN_EMB_WAVELENGTH,
    )
    # optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    # learning rate scheduler
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=15,
        cycle_mult=2,
        max_lr=LEARNING_RATE,
        min_lr=1e-6,
        warmup_steps=5,
        gamma=0.9
    )
    model.to(device)

    # If continuing training from checkpoint
    if args.pretrained_ckpt: 
        # Load checkpoint onto correct rank
        checkpoint = torch.load(args.pretrained_ckpt, map_location=device)
        consume_prefix_in_state_dict_if_present(checkpoint['model_state_dict'], "module.")
        model.load_state_dict(checkpoint['model_state_dict'])
        # Load optimizer
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Load scheduler
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        cur_epoch = checkpoint['epoch']
    else:
        cur_epoch = 0
    model = DDP(model, device_ids=[device], output_device=device)

    print("Loaded model...")
    if PRED_CONTINUOUS:
        loss_fn = MSEloss
    else:
        loss_fn = nn.CrossEntropyLoss(ignore_index = PAD_TOKEN_ID, reduction='mean').to(device)
    softmax = nn.Softmax(dim=-1)

    dist.barrier()
    writer = SummaryWriter(os.path.join(ckpt_dir, 'tensorboard'))
    for i in range(cur_epoch + 1, EPOCHS + 1):
        train_loader.sampler.set_epoch(i)
        model.train()
        dist.barrier()
        running_loss = 0.0
        cum_acc = 0.0
        cum_impute_error = 0.0
        for index, data in tqdm(enumerate(train_loader)):
            index += 1
            data = data.to(device)
            data, labels = data_mask(data)
            if index % GRADIENT_ACCUMULATION != 0:
                with model.no_sync():
                    logits = model(data) #should be size batch_size x seq_len x num_bins (if PRED_CONTINUOUS: batch_size x seq_len x 1)
                    loss = loss_fn(logits.transpose(1, 2).squeeze(dim=1), labels) / GRADIENT_ACCUMULATION #squeeze needed for MSEloss, shouldn't affect x-ent loss
                    loss.backward()
            if index % GRADIENT_ACCUMULATION == 0:
                logits = model(data)
                loss = loss_fn(logits.transpose(1, 2).squeeze(dim=1), labels) / GRADIENT_ACCUMULATION
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e2))
                optimizer.step()
                optimizer.zero_grad()
            running_loss += loss.item()
            if PRED_CONTINUOUS:
                final = logits.squeeze()
                impute_error = ((labels != PAD_TOKEN_ID) * torch.abs(final - labels)).sum(dim=-1)
                cum_impute_error += impute_error.median().item() #keep track of median imputation error per batch; report avg across batches in epoch
            else: #calculating 0-1 accuracy only applies with categorical preds
                final = softmax(logits)[..., 1:-1]
                final = final.argmax(dim=-1) + 1
                pred_num = (labels != PAD_TOKEN_ID).sum(dim=-1)
                correct_num = ((labels != PAD_TOKEN_ID) * (final == labels)).sum(dim=-1)
                cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
        if PRED_CONTINUOUS:
            epoch_impute_error = cum_impute_error / index
            epoch_impute_error = get_reduced(epoch_impute_error, device, 0, world_size)
            epoch_acc =-1
        else:
            epoch_acc = 100 * cum_acc / index
            epoch_acc = get_reduced(epoch_acc, device, 0, world_size)
            epoch_abs_error = -1
        epoch_loss = running_loss / index
        epoch_loss = get_reduced(epoch_loss, device, 0, world_size)
        if is_master:
            if PRED_CONTINUOUS:
                print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}% | Median Imputation Error : {epoch_impute_error:.4f} ==')
            else:
                print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}% ==')
        dist.barrier()
        scheduler.step()

        if i % VALIDATE_EVERY == 0:
            model.eval()
            dist.barrier()
            running_loss = 0.0
            running_error = 0.0
            predictions = []
            truths = []
            with torch.no_grad():
                for index, data in enumerate(val_loader):
                    index += 1
                    data = data.to(device)
                    data, labels = data_mask(data)
                    logits = model(data)
                    loss = loss_fn(logits.transpose(1, 2).squeeze(dim=1), labels)
                    running_loss += loss.item()
                    softmax = nn.Softmax(dim=-1)
                    if PRED_CONTINUOUS:
                        final = logits.squeeze(dim=2) #(include 'dim' in case at the end of loop, batchsize=1)
                    else:
                        final = softmax(logits)[..., 1:-1]
                        final = final.argmax(dim=-1) + 1
                    predictions.append(final)
                    truths.append(labels)
                del data, labels, logits, final
                # gather
                predictions = distributed_concat(torch.cat(predictions, dim=0), len(val_sampler.dataset), world_size)
                truths = distributed_concat(torch.cat(truths, dim=0), len(val_sampler.dataset), world_size)
                val_num = (truths != PAD_TOKEN_ID).sum(dim=-1)

                # Epoch loss
                val_loss = running_loss / index
                val_loss = get_reduced(val_loss, device, 0, world_size)

                # accuracy (categorical output) or absolute error (continuous output)
                if PRED_CONTINUOUS:
                    val_impute_error = ((truths != PAD_TOKEN_ID) * torch.abs(predictions - truths)).sum(dim=-1).median().item()
                    val_acc = -1 
                else:
                    correct_num = ((truths != PAD_TOKEN_ID) * (predictions == truths)).sum(dim=-1)
                    val_acc = 100 * (correct_num / val_num).mean().item()
                    val_impute_error = -1

            if is_master:
                if PRED_CONTINUOUS:
                    print(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.6f} | Accuracy: {val_acc:6.4f}% | Median Imputation Error: {val_impute_error:.4f} ==')
                else:
                    print(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.6f} | Accuracy: {val_acc:6.4f}% ==')

                duration = time() - cur_time
                cur_time = time()

                writer.add_scalar('Epoch duration', duration, i)
                writer.add_scalar('Loss/val', val_loss, i)
                writer.add_scalar('Loss/val', val_loss, i)
                if PRED_CONTINUOUS:
                    writer.add_scalar('Median imputation error/train', epoch_impute_error, i)
                    writer.add_scalar('Median imputation error/val', val_impute_error, i)
                else:
                    writer.add_scalar('Accuracy/train', epoch_acc, i)
                    writer.add_scalar('Accuracy/val', val_acc, i)

        del predictions, truths
        if is_master:
            save_ckpt(i, model, optimizer, scheduler, epoch_loss, model_name, ckpt_dir)
    
    cleanup()


def setup_process(rank, master_addr, master_port, world_size, backend="nccl"):
    print(f"Setting up process: rank={rank} world_size={world_size} backend={backend}.")
    print(f"master_addr={master_addr} master_port={master_port}")
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


if __name__=="__main__":
    main()
