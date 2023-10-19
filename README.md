# sc-foundation-eval
Code for evaluating single cell foundation models scBERT and scGPT. This code was used for the analysis presented in A Deep Dive into Single-Cell RNA Sequencing Foundation Models, bioRxiv.

The repo is organized by scripts and analysis code used to analyze each model.

## scBERT
* [performer_pytorch/](scBERT/performer_pytorch) contains the code for the scBERT model
* preprocess.py is a script provided by the scBERT authors, used to preprocess a dataset for fine-tuning
* dist_pretrain.py: used to pre-train scBERT from scratch
* dist_finetune.py: used to run fine-tuning (cell type annotation) for scBERT (Table 1). For our "no gene2vec" ablation (Table 2), do not pass the argument `--pos_embed_g2v` when calling this script.
  * An example command line call to run fine-tuning: `python dist_finetune.py --model_name finetune_seed2021 --data_path <path to preprocessed h5ad for fine-tuning> --model_path <path to pre-trained model> --world_size=1 --seed=2021 --epochs=10 --grad_acc=1 --batch_size=32 --pos_embed_g2v`
* dist_finetune_nopretraining.py: run our "no pre-training" ablation on scBERT (Table 2)
  * Similar command line call as above, but you do not need to supply a model_path, since this script does not load a pre-trained model (if you do supply one, it will be ignored and the ablation will still run properly)
* scbert_baselines_LR.ipynb shows example code for running the logistic regression baseline for annotating cell types in the Zheng68K PBMC dataset, including the few-shot setting

## scGPT
* scGPT_baselines_LR.py: runs the logistic regression baseline for annotating cell types in the myeloid, multiple sclerosis, and pancreas datasets, including the few-shot settings
* scGPT_run_all_celltypeannot_fewshot.py: runs scGPT fine-tuning for annotating cell types in the myeloid, multiple sclerosis, and pancreas datasets, including the few-shot settings. Based on the [annotation tutorial](tutorials/Tutorial_Annotation.ipynb) provided in scGPT's GitHub repo.
* scGPT_run_all_celltypeannot_nopretrain{_freeze}.py: run our "no pre-training" ablation on scGPT, with or without freezing pre-decoder weights (Supp. Figure 6, Supp. Table 5)
* create_figures_and_tables.ipynb: take the output of the previous scripts to create Figure 3, Supp. Figure 6, and Supp. Table 5
