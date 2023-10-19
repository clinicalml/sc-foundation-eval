# -*- coding: utf-8 -*-
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
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
import torch
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm

from performer_pytorch import PerformerLM
import scanpy as sc
import anndata as ad
from utils import *
from datetime import datetime
from time import time
import torch.multiprocessing as mp
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.utils.tensorboard import SummaryWriter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_addr", type=str, default="127.0.0.1", help='Master addr for dist finetune.')
    parser.add_argument("--master_port", type=str, default="8500", help='Master port for dist finetune.')
    parser.add_argument("--world_size", type=int, default=1, help='Number of GPUs.')
    parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
    parser.add_argument("--gene_num", type=int, default=None, help='Number of genes.') # 16906, if not supplied, will take the number of genes in the supplied training data
    parser.add_argument("--epochs", type=int, default=10, help='Number of epochs.')
    parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
    parser.add_argument("--batch_size", type=int, default=32, help='Number of batch size.')
    parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
    parser.add_argument("--grad_acc", type=int, default=1, help='Number of gradient accumulation.')
    parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
    parser.add_argument("--pos_embed_g2v", action='store_true', help='Using Gene2vec encoding or not (default no unless this arg is supplied).')
    parser.add_argument("--g2v_file", type=str, default='/data/rna_rep_learning/scBERT/gene2vec_16906.npy', help='File containing Gene2vec embeddings')
    parser.add_argument("--data_path", type=str, default='/data/rna_rep_learning/scBERT/Zheng68K.h5ad', help='Path of data for finetune.')
    parser.add_argument("--model_path", type=str, default='ckpts/panglao_full_with_g2v/2022-May-11-17:38:47/panglao_full_with_g2v_epoch_17.pth', help='Path of pretrained checkpoint to load.')
    parser.add_argument("--ft_ckpt", action="store_true", help="Add this flag if continuing to train an already finetuned model.")
    parser.add_argument("--ckpt_dir", type=str, default='./ckpts/', help='Directory for saving checkpoints.')
    parser.add_argument("--nreps", type=int, default=3, help='Number of replicates for each data split experiment.')
    #parser.add_argument("--sampling_fracs", type=list, default=[1.0, 0.75, 0.5, 0.25, 0.1], help='List of fractions of training data to sample for sample efficiency experiments.') #passing a list doesn't actually work
    parser.add_argument("--debug", action="store_true", help="Debug setting: saves to new dir.")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%b-%d-%H:%M:%S")
    
    mp.spawn(
        distributed_finetune,
        args=(args, timestamp),
        nprocs=args.world_size,
        join=True,
    )


def distributed_finetune(rank, args, timestamp):

    SEED = args.seed
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    GRADIENT_ACCUMULATION = args.grad_acc
    LEARNING_RATE = args.learning_rate
    VALIDATE_EVERY = args.valid_every
    CLASS = args.bin_num + 2
    POS_EMBED_USING = args.pos_embed_g2v
    PATIENCE = 10
    UNASSIGN_THRES = 0.0
    NREPS = args.nreps
    SAMPLING_FRACS = [1.0, 0.75, 0.5, 0.25, 0.1] #arg doesn't work currently

    is_master = rank == 0
    master_addr = args.master_addr
    master_port = args.master_port
    world_size = args.world_size

    ### CLASSES FROM ORIGINAL CODE ###

    class SCDataset(Dataset):
        def __init__(self, data, label):
            super().__init__()
            self.data = data
            self.label = label

        def __getitem__(self, index):
            #rand_start = random.randint(0, self.data.shape[0]-1)
            full_seq = self.data[index].toarray()[0]
            full_seq[full_seq > (CLASS - 2)] = CLASS - 2
            full_seq = torch.from_numpy(full_seq).long() #long() converts to int64
            full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device) #this is the CLS token ?
            seq_label = self.label[index]
            return full_seq, seq_label

        def __len__(self):
            return self.data.shape[0]

    class Identity(torch.nn.Module):
        def __init__(self, dropout = 0., h_dim = 100, out_dim = 10):
            super(Identity, self).__init__()
            self.conv1 = nn.Conv2d(1, 1, (1, 200))
            self.act = nn.ReLU()
            self.fc1 = nn.Linear(in_features=SEQ_LEN, out_features=512, bias=True)
            self.act1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout)
            self.fc2 = nn.Linear(in_features=512, out_features=h_dim, bias=True)
            self.act2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout)
            self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

        def forward(self, x):
            x = x[:,None,:,:]
            x = self.conv1(x)
            x = self.act(x)
            x = x.view(x.shape[0],-1)
            x = self.fc1(x)
            x = self.act1(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            x = self.act2(x)
            x = self.dropout2(x)
            x = self.fc3(x)
            return x

    cur_time = time()
    setup_process(rank, master_addr, master_port, world_size)
    device = torch.device("cuda:{}".format(rank))

    print("Set up distributed processes...")

    data = sc.read_h5ad(args.data_path)
    if args.debug:
        debug_seq_len = 5000
        data = data[:1000,:debug_seq_len]
        GRADIENT_ACCUMULATION = 1
    label_dict, label = np.unique(np.array(data.obs['celltype']), return_inverse=True)  # Convert strings categorical to integrate categorical, and label_dict[label] can be restored
    class_num = np.unique(label, return_counts=True)[1].tolist() 
    #class_weight = torch.tensor([(1 - (x / sum(class_num))) ** 2 for x in class_num]) #doesn't get used anywhere
    class_weight = torch.tensor([1/x for x in class_num])  #use this simpler weighting
    label = torch.from_numpy(label)
    data = data.X
    if args.gene_num is not None:
        SEQ_LEN = args.gene_num + 1
    else:
        SEQ_LEN = data.shape[1] + 1 # num_genes + 1


    def instantiate_new_model():
    #create new model
        model = PerformerLM(
            num_tokens = CLASS,
            dim = 200,
            depth = 6,
            max_seq_len = SEQ_LEN,
            heads = 10,
            local_attn_heads = 0,
            g2v_position_emb = POS_EMBED_USING,
            g2v_file = args.g2v_file
        )
        model = model.to(device)

        # Load checkpoint onto correct rank
        checkpoint = torch.load(args.model_path, map_location=device)
        consume_prefix_in_state_dict_if_present(checkpoint['model_state_dict'], "module.")
        if args.ft_ckpt:
            print("Loaded finetuned ckpt...")
            model.to_out = Identity(dropout=0., h_dim=128, out_dim=label_dict.shape[0])
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)    
            cur_epoch = checkpoint['epoch']
            # Load optimizer
            #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Load scheduler
            #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            print("Loaded pretrained model...")
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to_out = Identity(dropout=0., h_dim=128, out_dim=label_dict.shape[0]).to(device)
            cur_epoch = 0

        for name, param in model.named_parameters():
            param.requires_grad = False
        for name, param in model.norm.named_parameters():
            param.requires_grad = True
        for name, param in model.performer.net.layers[-1].named_parameters(): #make last layers of performer trainable during fine tuning
            param.requires_grad = True
        for name, param in model.to_out.named_parameters():
            param.requires_grad = True

        # optimizer
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=15,
            cycle_mult=2,
            max_lr=LEARNING_RATE,
            min_lr=1e-6,
            warmup_steps=5,
            gamma=0.9
        )

        return(model, optimizer, scheduler, cur_epoch)
                
    try:
        for k in np.arange(NREPS):
            # Control sources of randomness - for each run k, different seed is used 
            # this effects parameter initialization & the subsampling of the [fixed] training set
            torch.manual_seed(SEED*k)
            random.seed(SEED*k)
            np.random.seed(SEED*k)

            accs = []
            f1s = []
            for frac in SAMPLING_FRACS:
                #create new model
                model, optimizer, scheduler, cur_epoch = instantiate_new_model()
                model = DDP(model, device_ids=[device], output_device=device)
        
                #ckpt dir setup, only need one process to create directory so use dist.barrier()
                model_name = "finetune_sampleeff_{}_{}".format(frac, k)
                ckpt_dir = os.path.join("ckpts/", "finetune-sampleeff-"+timestamp, model_name)
                if is_master:
                    print("Checkpoint dir: ", ckpt_dir)
                    if not (os.path.exists(ckpt_dir)):
                        os.makedirs(ckpt_dir)
                dist.barrier()


                #implement class weights in loss to handle class imbalance
                loss_fn = nn.CrossEntropyLoss(weight=class_weight).to(device)

                dist.barrier()
                trigger_times = 0
                max_acc = 0.0
                writer = SummaryWriter(os.path.join(ckpt_dir, 'tensorboard'))

                # attempt to seed dataloader - this is required for true reproducibility
                def seed_worker(worker_id):
                    worker_seed = torch.initial_seed() % 2**32
                    numpy.random.seed(worker_seed)
                    random.seed(worker_seed)

                g = torch.Generator()
                g.manual_seed(0)

                #downsample training set
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2022) #same val set across all runs
                for index_train, index_val in sss.split(data, label):
                    index_train_small = np.random.choice(index_train, round(index_train.shape[0]*frac), replace=False) # different random subset will be chosen with each k
                    data_train, label_train = data[index_train_small], label[index_train_small]
                    train_dataset = SCDataset(data_train, label_train)
                    data_val, label_val = data[index_val], label[index_val]
                    val_dataset = SCDataset(data_val, label_val)
                train_sampler = DistributedSampler(train_dataset, shuffle=True)
                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, worker_init_fn=seed_worker, generator=g)
                val_sampler = DistributedSampler(val_dataset, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, worker_init_fn=seed_worker, generator=g)

                print("Loaded data...")

                for i in range(cur_epoch+1, EPOCHS+1):
                    print("{} iterations in train dataloader per epoch".format(len(train_loader)))
                    train_loader.sampler.set_epoch(i)
                    model.train()
                    dist.barrier()
                    running_loss = 0.0
                    cum_acc = 0.0
                    for index, (data_t, labels_t) in tqdm(enumerate(train_loader)):
                        index += 1
                        data_t, labels_t = data_t.to(device), labels_t.to(device)
                        if index % GRADIENT_ACCUMULATION != 0:
                            with model.no_sync():
                                logits = model(data_t)
                                loss = loss_fn(logits, labels_t)
                                loss.backward()
                        if index % GRADIENT_ACCUMULATION == 0:
                            logits = model(data_t)
                            loss = loss_fn(logits, labels_t)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
                            optimizer.step()
                            optimizer.zero_grad()
                        running_loss += loss.item()
                        softmax = nn.Softmax(dim=-1)
                        final = softmax(logits)
                        final = final.argmax(dim=-1)
                        pred_num = labels_t.size(0)
                        correct_num = torch.eq(final, labels_t).sum(dim=-1)
                        cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
                    epoch_loss = running_loss / index
                    epoch_acc = 100 * cum_acc / index
                    epoch_loss = get_reduced(epoch_loss, device, 0, world_size)
                    epoch_acc = get_reduced(epoch_acc, device, 0, world_size)
                    if is_master:
                        print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}%  ==')
                    dist.barrier()
                    scheduler.step()

                    if i % VALIDATE_EVERY == 0:
                        model.eval()
                        dist.barrier()
                        running_loss = 0.0
                        predictions = []
                        truths = []
                        with torch.no_grad():
                            for index, (data_v, labels_v) in enumerate(val_loader):
                                index += 1
                                data_v, labels_v = data_v.to(device), labels_v.to(device)
                                logits = model(data_v)
                                loss = loss_fn(logits, labels_v)
                                running_loss += loss.item()
                                softmax = nn.Softmax(dim=-1)
                                final_prob = softmax(logits)
                                final = final_prob.argmax(dim=-1)
                                final[np.amax(np.array(final_prob.cpu()), axis=-1) < UNASSIGN_THRES] = -1
                                predictions.append(final)
                                truths.append(labels_v)
                            del data_v, labels_v, logits, final_prob, final
                            # gather
                            predictions = distributed_concat(torch.cat(predictions, dim=0), len(val_sampler.dataset), world_size)
                            truths = distributed_concat(torch.cat(truths, dim=0), len(val_sampler.dataset), world_size)
                            no_drop = predictions != -1
                            predictions = np.array((predictions[no_drop]).cpu())
                            truths = np.array((truths[no_drop]).cpu())
                            cur_acc = accuracy_score(truths, predictions)
                            f1 = f1_score(truths, predictions, average='macro')
                            val_loss = running_loss / index
                            val_loss = get_reduced(val_loss, device, 0, world_size)
                            if is_master:
                                print(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.6f} | F1 Score: {f1:.6f} | Accuracy: {cur_acc:.3f} ==')
                                print(confusion_matrix(truths, predictions))
                                print(classification_report(truths, predictions, labels=np.arange(len(label_dict)), target_names=label_dict.tolist(), digits=4))

                                duration = time() - cur_time
                                cur_time = time()

                                writer.add_scalar('Loss/train', epoch_loss, i)
                                writer.add_scalar('Accuracy/train', epoch_acc, i)
                                writer.add_scalar('Loss/val', val_loss, i)
                                writer.add_scalar('Accuracy/val', cur_acc, i)
                                writer.add_scalar('F1/val', f1, i)
                            if cur_acc > max_acc:
                                max_acc = cur_acc
                                trigger_times = 0
                                save_best_ckpt(i, model, optimizer, scheduler, val_loss, model_name, ckpt_dir)
                            else:
                                trigger_times += 1
                                if trigger_times > PATIENCE:
                                    break
                    del predictions, truths
                accs.append(cur_acc)
                f1s.append(f1)
            if is_master:
                print("fraction of training set:")
                print(SAMPLING_FRACS)
                print("effective fraction of full dataset:")
                print([np.round(s*0.8,2) for s in SAMPLING_FRACS]) #size of training set as fraction of overall dataset size
                print(accs)
                print(f1s)
                with open('logs/finetune_sampleeff_{}_{}.txt'.format(k, timestamp), 'a') as fd:
                    fd.write(','.join([str(a) for a in SAMPLING_FRACS])+'\n')
                    fd.write(','.join([str(np.round(s*0.8,2)) for s in SAMPLING_FRACS])+'\n')
                    fd.write(','.join(map(lambda x: str(x), accs))+'\n')
                    fd.write(','.join(map(lambda x: str(x), f1s))+'\n')
    except Exception as e:
        print(e)
        pass #so that cleanup() occurs with or without error
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
