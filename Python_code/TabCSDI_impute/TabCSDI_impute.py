import argparse
import torch
import datetime
import json
import yaml
import os
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np

import pdb
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

from src.main_model_table_ft import TabCSDI
from src.utils_table import train, evaluate_ft

config = "config_TabCSDI.yaml"

# check cuda
if torch.cuda.is_available():
    device = f"cuda"
else:
    device = "cpu"
print(device)

seed = 42
nfold = 5
unconditional = False
nsample = 100
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

## NEED TO CHANGE THE PATH BASED ON YOUR COMPUTER ##
path = "/home/cs-sun2/CMIE_Project/Python_code/TabCSDI_impute/parameters/" + config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = unconditional


def round_to_nearest_half(value):
    """
    Rounds a float value to the nearest 0.5

    Args:
        value (float): The value to round

    Returns:
        float: The value rounded to the nearest 0.5
    """
    return round(value * 2) / 2


def round_int(series, cate_num):
    """
    Maps values in a pandas Series based on conditions:
    - Values < 0 become 0
    - Values between [0, cate_num) rounded to closest integer
    - Values >= cate_num become cate_num
    """
    rounded_series = np.round(series)
    rounded_series[rounded_series <= 0] = 0
    rounded_series[rounded_series >= cate_num] = cate_num
    return rounded_series.astype(int)


class tabular_Dataset(Dataset):
    # eval_length should be equal to attributes number.
    def __init__(
        self,
        base_path,
        cohort,
        missing_method,
        missing_ratio,
        index_file,
        full_data,
        new_cols_order,
        cont_list,
        eval_length=24,
        use_index_list=None,
        seed=0,
        mode="train",
    ):
        self.eval_length = eval_length
        np.random.seed(seed)

        manully_mask_ratio = 0.1

        mask_path = (
            base_path
            / f"data_miss_mask/{cohort}/{cohort}_{mode}/{missing_method}/miss{missing_ratio}/{index_file}.csv"
        )

        mask_data = pd.read_csv(mask_path)
        mask_data = mask_data.reindex(columns=new_cols_order)
        self.observed_values = (
            full_data.mask(mask_data.astype(bool), 0).astype(float).values
        )
        self.observed_masks = (~mask_data.astype(bool)).values

        masks = self.observed_masks.copy()
        # In this section, obtain gt_masks
        # for each column, mask `missing_ratio` % of observed values.
        for col in range(masks.shape[1]):
            obs_indices = np.where(masks[:, col])[0]
            miss_indices = np.random.choice(
                obs_indices, (int)(len(obs_indices) * manully_mask_ratio), replace=False
            )
            masks[miss_indices, col] = False
        # gt_mask: 0 for missing elements and manully maksed elements
        self.gt_masks = masks.reshape(self.observed_masks.shape)

        self.cont_cols = cont_list

        print("--------Dataset created--------")

        self.use_index_list = np.arange(len(self.observed_values))

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


cohorts = ["C19"]
missing_method = ["MCAR", "MAR", "MNAR"]
missing_ratio = [10, 20, 30, 40, 50]
SampleTime = 5

## NEED TO CHANGE THE PATH BASED ON YOUR COMPUTER ##
# Define paths
base_path = Path("/Users/siysun/Desktop/PhD/SynthCPHS_benchmark/data_stored")

## NEED TO CHANGE THE PATH BASED ON YOUR COMPUTER ##
# Output path for time recording
output_file = Path(
    "/Users/siysun/Desktop/PhD/SynthCPHS_benchmark/imputation_times_tabcsdi.csv"
)
time_records = []

for cohort in cohorts:
    train_full_path = base_path / f"Completed_data/{cohort}/{cohort}_train.csv"
    test_full_path = base_path / f"Completed_data/{cohort}/{cohort}_test.csv"
    train_full = pd.read_csv(train_full_path)
    test_full = pd.read_csv(test_full_path)

    # preprocessing
    scaler = MinMaxScaler()
    con_col = [col for col in train_full.columns if col.startswith("con_")]
    train_full[con_col] = scaler.fit_transform(train_full[con_col])
    test_full[con_col] = scaler.transform(test_full[con_col])

    # encode categorical feature
    encoders = {}
    for col in train_full.columns:
        if col.startswith("cat_"):
            tem_encoder = LabelEncoder()
            train_full[col] = tem_encoder.fit_transform(train_full[col])
            test_full[col] = tem_encoder.transform(test_full[col])
            encoders[col] = tem_encoder

    # Reorder columns (continuous first, then categorical)
    cat_list = [col for col in train_full.columns if col.startswith("cat_")]
    con_list = [col for col in train_full.columns if col.startswith("con_")]
    new_cols_order = con_list + cat_list
    train_full_data = train_full.reindex(columns=new_cols_order)
    test_full_data = test_full.reindex(columns=new_cols_order)
    cont_cols = [i for i in range(len(con_list))]
    num_cate_list = [train_full[col].nunique() for col in cat_list]

    for method in missing_method:
        for ratio in missing_ratio:
            # Define paths and create output directory
            train_save_dir = (
                f"data_tabcsdi/{cohort}/{cohort}_train/{method}/miss{ratio}"
            )
            test_save_dir = f"data_tabcsdi/{cohort}/{cohort}_test/{method}/miss{ratio}"
            train_output_dir = base_path / train_save_dir
            test_output_dir = base_path / test_save_dir
            train_output_dir.mkdir(parents=True, exist_ok=True)
            test_output_dir.mkdir(parents=True, exist_ok=True)

            imputation_times = []  # Store times for averaging

            for index_file in tqdm(range(SampleTime)):
                save_train_path = train_output_dir / f"{index_file}.csv"
                save_test_path = test_output_dir / f"{index_file}.csv"

                batch_size = config["train"]["batch_size"]
                feature_dim = config["model"]["featuredim"]

                # def get_dataloader(seed=1, nfold=5, batch_size=16, missing_ratio=0.1):
                train_dataset = tabular_Dataset(
                    base_path,
                    cohort,
                    method,
                    ratio,
                    index_file,
                    train_full_data,
                    new_cols_order,
                    cont_cols,
                    feature_dim,
                    seed,
                    mode="train",
                )
                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=1
                )

                # def get_dataloader(seed=1, nfold=5, batch_size=16, missing_ratio=0.1):
                test_dataset = tabular_Dataset(
                    base_path,
                    cohort,
                    method,
                    ratio,
                    index_file,
                    test_full_data,
                    new_cols_order,
                    cont_cols,
                    feature_dim,
                    seed,
                    mode="test",
                )
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)

                print(f"Training dataset size: {len(train_dataset)}")
                print(f"Testing dataset size: {len(test_dataset)}")

                model = TabCSDI(
                    "cmie",
                    config,
                    device,
                    cont_cols,
                    num_cate_list,
                    target_dim=1,
                    base_path=base_path,
                ).to(device)

                # Add timing measurement
                start_time = time.time()

                ################### Train #########################
                model = train(
                    model,
                    config["train"],
                    train_loader,
                    valid_loader=None,
                    foldername="",
                )

                ################### Imputation #########################
                out_train = evaluate_ft(model, train_loader, nsample=5)
                out_train_df = pd.DataFrame(out_train, columns=new_cols_order)[
                    train_full.columns
                ]
                out_train_df[con_col] = scaler.inverse_transform(out_train_df[con_col])

                # decode categorical
                for col in out_train_df.columns:
                    if col.startswith("cat_"):
                        tem_encoder = encoders[col]
                        cate_num = len(tem_encoder.classes_)
                        out_train_df[col] = round_int(out_train_df[col], cate_num - 1)
                        out_train_df[col] = tem_encoder.inverse_transform(
                            out_train_df[col]
                        )
                    elif col.startswith("con_TS_ON"):
                        out_train_df[col] = round_to_nearest_half(out_train_df[col])
                    else:
                        out_train_df[col] = round(out_train_df[col])

                # imputation time
                end_time = time.time()
                imputation_time = end_time - start_time
                imputation_times.append(imputation_time)

                out_train_df.to_csv(save_train_path, index=False)

                out_test = evaluate_ft(model, test_loader, nsample=5)
                out_test_df = pd.DataFrame(out_test, columns=new_cols_order)[
                    train_full.columns
                ]
                out_test_df[con_col] = scaler.inverse_transform(out_test_df[con_col])

                # decode categorical
                for col in out_test_df.columns:
                    if col.startswith("cat_"):
                        tem_encoder = encoders[col]
                        cate_num = len(tem_encoder.classes_)
                        out_test_df[col] = round_int(out_test_df[col], cate_num - 1)
                        out_test_df[col] = tem_encoder.inverse_transform(
                            out_test_df[col]
                        )
                    elif col.startswith("con_TS_ON"):
                        out_test_df[col] = round_to_nearest_half(out_test_df[col])
                    else:
                        out_test_df[col] = round(out_test_df[col])

                out_test_df.to_csv(save_test_path, index=False)

                # Compute average time
                avg_time = sum(imputation_times) / len(imputation_times)

                # Store result
                time_records.append(
                    [cohort, "TabCSDI", method, ratio, index_file, avg_time]
                )

                # Save results to CSV
                time_df = pd.DataFrame(
                    time_records,
                    columns=[
                        "Dataset",
                        "Method",
                        "Mechanism",
                        "MissingRatio",
                        "Index file",
                        "AvgTime",
                    ],
                )
                time_df.to_csv(output_file, index=False)
                print(f"{method}, {ratio} Imputation times recorded.")
