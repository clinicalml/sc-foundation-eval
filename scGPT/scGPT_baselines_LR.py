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
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score
from tqdm import tqdm

import scanpy as sc
import anndata as ad
from datetime import datetime
from time import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from pathlib import Path

# Define the file path where you want to store the results
res_file_path = 'LR_sampleeff_results.txt'
with open(res_file_path, 'a') as file:
    file.write("dataset_name\tfraction\tseed\tc\taccuracy\tprecision\trecall\tmacro_f1\n")

now = datetime.now()

for DATASET_NAME in ['ms', 'pancreas', 'myeloid']:
    print(DATASET_NAME)            
    ## Step 2: Load and pre-process data

    if DATASET_NAME == "ms":
        data_dir = Path("/localdata/rna_rep_learning/scGPT/ms") #RB
        adata = sc.read(data_dir / "c_data.h5ad")
        adata_test = sc.read(data_dir / "filtered_ms_adata.h5ad")
        adata.obs["celltype"] = adata.obs["Factor Value[inferred cell type - authors labels]"].astype("category")
        adata_test.obs["celltype"] = adata_test.obs["Factor Value[inferred cell type - authors labels]"].astype("category")
        adata.obs["batch_id"]  = adata.obs["str_batch"] = "0"
        adata_test.obs["batch_id"]  = adata_test.obs["str_batch"] = "1"          
        adata.var.set_index(adata.var["gene_name"], inplace=True)
        adata_test.var.set_index(adata.var["gene_name"], inplace=True)
        data_is_raw = False
        filter_gene_by_counts = False
        adata_test_raw = adata_test.copy()
        adata = adata.concatenate(adata_test, batch_key="str_batch")

    if DATASET_NAME == "pancreas":
        data_dir = Path("/localdata/rna_rep_learning/scGPT/pancreas")
        adata = sc.read(data_dir / "demo_train.h5ad")
        adata_test = sc.read(data_dir / "demo_test.h5ad")
        adata.obs["celltype"] = adata.obs["Celltype"].astype("category")
        adata_test.obs["celltype"] = adata_test.obs["Celltype"].astype("category")
        adata.obs["batch_id"]  = adata.obs["str_batch"] = "0"
        adata_test.obs["batch_id"]  = adata_test.obs["str_batch"] = "1"    
        data_is_raw = False
        filter_gene_by_counts = False   
        adata_test_raw = adata_test.copy()
        adata = adata.concatenate(adata_test, batch_key="str_batch")

    if DATASET_NAME == "myeloid":
        data_dir = Path("/localdata/rna_rep_learning/scGPT/myeloid/")
        adata = sc.read(data_dir / "reference_adata.h5ad")
        adata_test = sc.read(data_dir / "query_adata.h5ad")
        adata.obs["celltype"] = adata.obs["cell_type"].astype("category")
        adata_test.obs["celltype"] = adata_test.obs["cell_type"].astype("category")
        adata.obs["batch_id"]  = adata.obs["str_batch"] = "0"
        adata_test.obs["batch_id"]  = adata_test.obs["str_batch"] = "1"          
        adata_test_raw = adata_test.copy()
        data_is_raw = False
        filter_gene_by_counts = False   
        adata = adata.concatenate(adata_test, batch_key="str_batch")

    # make the batch category column
    batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
    adata.obs["batch_id"] = batch_id_labels
    celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
    celltypes = adata.obs["celltype"].unique()
    num_types = len(np.unique(celltype_id_labels))
    id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
    adata.obs["celltype_id"] = celltype_id_labels
    adata.var["gene_name"] = adata.var.index.tolist()

    adata_test = adata[adata.obs["str_batch"] == "1"]
    adata = adata[adata.obs["str_batch"] == "0"]

    all_counts_full = (
                adata.X.A
                if sparse.issparse(adata.X)
                else adata.X
            )
    all_counts_test = (
                adata_test.X.A
                if sparse.issparse(adata_test.X)
                else adata_test.X
            )
    genes = adata.var["gene_name"].tolist()

    celltypes_labels_full = adata.obs["celltype_id"].tolist()  # make sure count from 0
    celltypes_labels_full = np.array(celltypes_labels_full)
    celltypes_labels_test = adata_test.obs["celltype_id"].tolist()  # make sure count from 0
    celltypes_labels_test = np.array(celltypes_labels_test)
    
    #batch_ids = adata.obs["batch_id"].tolist()
    #num_batch_types = len(set(batch_ids))
    #batch_ids = np.array(batch_ids)
    
    for FRAC in [0.1, 0.25, 0.5, 0.75, 1]:
        for SEED in [1,2,3,4,5,6,7,8,9,10]:
            print("fraction: ", FRAC)
            print("SEED: ", SEED)
            save_dir = Path(f"./save/logisticregression/lr-{DATASET_NAME}-frac{FRAC}-seed{SEED}-{now}/")
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"save to {save_dir}")

            ## optionally subset train data for few shot experiments - RB
            if FRAC != 1:
                print("subsetting to {}% training data".format(FRAC*100))
                all_counts, _, celltypes_labels, _, = train_test_split(all_counts_full, celltypes_labels_full, train_size=FRAC, random_state=SEED, shuffle=True, stratify=celltypes_labels_full)
            else:
                all_counts = all_counts_full.copy()
                celltypes_labels = celltypes_labels_full.copy()

            ## choose c using k fold cross val 
            #if SEED == 1: # can do this just once per fraction (share c across seeds)
            print("running cross validation to choose c...")
            cv_results = {}
            for c in [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]:
                lr = LogisticRegression(random_state=0, penalty="l1", C=c, solver="liblinear")
                res = cross_validate(lr, all_counts, celltypes_labels, scoring=['accuracy'])
                cv_results[c] = np.mean(res['test_accuracy'])

            #choose best c 
            best_ind = np.argmax(list(cv_results.values()))
            c = list(cv_results.keys())[best_ind]
            print(f'for {FRAC*100}% of {DATASET_NAME} data with seed {SEED}, best c is {c}')
                
            ## run LR
            lr = LogisticRegression(penalty="l1", C=c, solver="liblinear", random_state=SEED)
            lr.fit(all_counts, celltypes_labels)

            test_acc = lr.score(all_counts_test, celltypes_labels_test)
            print("test set accuracy: " + str(np.around(test_acc, 4)))

            test_recall = recall_score(celltypes_labels_test, lr.predict(all_counts_test), average="macro") #Based on github, looks like they actually used macro (they dont claim to in paper, but is consistent with their results)
            print("test set recall: " + str(np.around(test_recall, 4)))

            test_precision = precision_score(celltypes_labels_test, lr.predict(all_counts_test), average="macro") #Based on github, looks like they actually used macro (they dont claim to in paper, but is consistent with their results)
            print("test set precision: " + str(np.around(test_precision, 4)))

            test_macro_f1 = f1_score(celltypes_labels_test, lr.predict(all_counts_test), average="macro")
            print("test set macro F1: " + str(np.around(test_macro_f1, 4)))
            
            ## plot confusion matrix
            from sklearn.metrics import confusion_matrix
            celltypes = list(celltypes)
            predictions = lr.predict(all_counts_test)
            for i in set([id2type[p] for p in predictions]):
                if i not in celltypes:
                    celltypes.remove(i)
                    print("removing cell type {}".format(i)) 
            cm = confusion_matrix(celltypes_labels_test, predictions)
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            cm = pd.DataFrame(cm, index=celltypes[:cm.shape[0]], columns=celltypes[:cm.shape[1]])
            plt.figure(figsize=(10, 10))
            sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues")
            plt.savefig(save_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")

            ## write results to file
            with open(res_file_path, 'a') as file:
                file.write(f"{DATASET_NAME}\t{FRAC}\t{SEED}\t{c}\t{test_acc}\t{test_precision}\t{test_recall}\t{test_macro_f1}\n")
            
            print("\n")
            