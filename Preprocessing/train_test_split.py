#!/home/siyi.sun/miniconda3/bin python3
# -*- coding: UTF-8 -*-
"""
@Description :   Split the train test dataset and romove target from the dataset
@Author      :   siyi.sun
@Time        :   2025/03/02 16:54:44
"""

from pathlib import Path
import pandas as pd
import numpy as np

## TODO: Add more data's target
INFO = {"SynthSurvey": "cat_educ_attain"}  # the target column name in the dataset

cohort = "SynthSurvey"
## NEED TO CHANGE THE PATH BASED ON YOUR COMPUTER ##
base_path = Path("/Users/siysun/Desktop/NeurIPS25/data_stored")
path = base_path / f"Completed_data/{cohort}/{cohort}_all.csv"
split_ratio = 0.8
# path to save the features(x) and labels(y)
train_x_save = base_path / f"Completed_data/{cohort}/{cohort}_train.csv"
test_x_save = base_path / f"Completed_data/{cohort}/{cohort}_test.csv"
train_y_save = base_path / f"Completed_data/{cohort}/{cohort}_train_y.csv"
test_y_save = base_path / f"Completed_data/{cohort}/{cohort}_test_y.csv"

# read the synthetic data
data_df = pd.read_csv(path)

# generate the train and test index
total_num = data_df.shape[0]
keep_idx = np.arange(total_num)

num_train = int(keep_idx.shape[0] * split_ratio)
num_test = total_num - num_train
seed = 1234

np.random.seed(seed)
np.random.shuffle(keep_idx)

train_idx = keep_idx[:num_train]
test_idx = keep_idx[-num_test:]
train_df_full = data_df.loc[train_idx]
test_df_full = data_df.loc[test_idx]

# save the full train and test X dataset
train_df_full.drop(columns=INFO[cohort]).to_csv(train_x_save, index=False)
test_df_full.drop(columns=INFO[cohort]).to_csv(test_x_save, index=False)

# save the full train and test Y dataset for downstream classification task
train_df_full[INFO[cohort]].to_csv(train_y_save, index=False)
test_df_full[INFO[cohort]].to_csv(test_y_save, index=False)
