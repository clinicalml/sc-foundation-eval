# sc-foundation-eval
Code for evaluating single cell foundation models scBERT and scGPT. This code was used for the analysis presented in A Deep Dive into Single-Cell RNA Sequencing Foundation Models, bioRxiv https://doi.org/10.1101/2023.10.19.563100.

The repo is organized by model. Below are descriptions of the scripts and analysis code included for each:

## scBERT
* [performer_pytorch/](scBERT/performer_pytorch) contains the code for the scBERT model
* preprocess.py is a script provided by the scBERT authors, used to preprocess a dataset for fine-tuning
* dist_pretrain.py: used to pre-train scBERT from scratch
* dist_finetune.py: used to run fine-tuning (cell type annotation) for scBERT (Table 1). For our "no gene2vec" ablation (Table 2), do not pass the argument `--pos_embed_g2v` when calling this script.
  * An example command line call to run fine-tuning: `python dist_finetune.py --model_name finetune_seed2021 --data_path <path to preprocessed h5ad for fine-tuning> --model_path <path to pre-trained model> --world_size=1 --seed=2021 --epochs=10 --grad_acc=1 --batch_size=32 --pos_embed_g2v`
* dist_finetune_nopretraining.py: run our "no pre-training" ablation on scBERT (Table 2)
  * Similar command line call as above, but you do not need to supply a model_path, since this script does not load a pre-trained model (if you do supply one, it will be ignored and the ablation will still run properly)
* dist_finetune_fewshot.py: run scBERT fine-tuning on 10, 25, 50, 75, and 100\% of the training data
* scbert_baselines_LR.ipynb shows example code for running the logistic regression baseline for annotating cell types in the Zheng68K PBMC dataset, including the few-shot setting
* nog2v_explore.ipynb: an exploration of pre-training performance for our "no gene2vec" ablation, including the results shown in Table 3
* collate_final_results_finetune.ipynb: collate results of fine-tuning scBERT (full and few-shot settings), logistic regression (full and few-shot settings), and ablation studies to create Tables 1 & 2 and Figure 2
  
## scGPT
* scGPT_baselines_LR.py: runs the logistic regression baseline for annotating cell types in the myeloid, multiple sclerosis, and pancreas datasets, including the few-shot settings
* scGPT_run_all_celltypeannot_fewshot.py: runs scGPT fine-tuning for annotating cell types in the myeloid, multiple sclerosis, and pancreas datasets, including the few-shot settings. Based on the [annotation tutorial](tutorials/Tutorial_Annotation.ipynb) provided in scGPT's GitHub repo.
* scGPT_run_all_celltypeannot_nopretrain{_freeze}.py: run our "no pre-training" ablation on scGPT, with or without freezing pre-decoder weights (Supp. Figure 6, Supp. Table 5)
* create_figures_and_tables.ipynb: take the output of the previous scripts to create Figure 3, Supp. Figure 6, and Supp. Table 5

## Data Availability

### scBERT datasets
* The Zheng68K PBMC data used for finetuning scBERT can be downloaded from our [data/](data) directory. It has been processed using the scBERT/preprocess.py script.
  * preprocess.py requires panglao_1000.h5ad, a subsampled version of the panglao dataset on which scBERT was pre-trained, also available in [data/](data).
* The full panglao dataset used for pretraining is too large to host on GitHub, but can be downloaded as per the [instructions](https://github.com/TencentAILabHealthcare/scBERT#data) from the scBERT authors.

### scGPT datasets
As provided by the scGPT authors:
- Multiple Sclerosis (M.S.) dataset: [link](https://drive.google.com/drive/folders/1Qd42YNabzyr2pWt9xoY4cVMTAxsNBt4v?usp=sharing)

- Myeloid (Mye.) dataset: [link](https://drive.google.com/drive/folders/1VbpApQufZq8efFGakW3y8QDDpY9MBoDS?usp=drive_link)

- hPancreas dataset: [link](https://drive.google.com/drive/folders/1s9XjcSiPC-FYV3VeHrEa7SeZetrthQVV?usp=drive_link)
