from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

from geomloss import SamplesLoss

from imputers import RRimputer

from utils import *

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.debug("test")

torch.set_default_tensor_type("torch.DoubleTensor")


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


## NEED TO CHANGE THE PATH BASED ON YOUR COMPUTER ##
base_path = Path("/Users/siysun/Desktop/PhD/SynthCPHS_benchmark/data_stored")
batchsize = 128  # If the batch size is larger than half the dataset's size,
# it will be redefined in the imputation methods.
lr = 1e-2

cohort = "C19"
methods = ["MCAR", "MAR", "MNAR"]
ratios = [50, 40, 30, 20, 10]
Sampletime = 5

## NEED TO CHANGE THE PATH BASED ON YOUR COMPUTER ##
# Output path for time recording
output_file = Path(
    f"/Users/siysun/Desktop/PhD/SynthCPHS_benchmark/imputation_times_MOT_Synthetic_C19.csv"
)
time_records = []

# read full dataset
full_train_path = base_path / f"Completed_data/{cohort}/{cohort}_train.csv"
full_test_path = base_path / f"Completed_data/{cohort}/{cohort}_test.csv"
train_full = pd.read_csv(full_train_path)
test_full = pd.read_csv(full_test_path)

# preprocessing
scaler = StandardScaler()
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

X_train_true = torch.from_numpy(train_full.to_numpy())

for method in tqdm(methods):
    for ratio in tqdm(ratios):
        save_dir_train = (
            base_path / f"data_MOT/{cohort}/{cohort}_train/{method}/miss{ratio}"
        )
        save_dir_test = (
            base_path / f"data_MOT/{cohort}/{cohort}_test/{method}/miss{ratio}"
        )
        save_dir_train.mkdir(parents=True, exist_ok=True)
        save_dir_test.mkdir(parents=True, exist_ok=True)

        imputation_times = []  # Store times for averaging

        for index_file in tqdm(range(Sampletime)):
            # path for mask dataset
            mask_train_path = (
                base_path
                / f"data_miss_mask/{cohort}/{cohort}_train/{method}/miss{ratio}/{index_file}.csv"
            )
            mask_test_path = (
                base_path
                / f"data_miss_mask/{cohort}/{cohort}_test/{method}/miss{ratio}/{index_file}.csv"
            )
            # path for save imputed dataset
            save_train_path = save_dir_train / f"{index_file}.csv"
            save_test_path = save_dir_test / f"{index_file}.csv"

            train_mask = pd.read_csv(mask_train_path)
            test_mask = pd.read_csv(mask_test_path)
            train_miss = torch.from_numpy(
                train_full.mask(train_mask.astype(bool)).to_numpy()
            )
            test_miss = torch.from_numpy(
                test_full.mask(test_mask.astype(bool)).to_numpy()
            )

            # hyperparameter
            n, d = train_miss.shape
            epsilon = pick_epsilon(train_miss)  # Set the regularization parameter a

            # Create the imputation models
            d_ = d - 1
            models = {}

            for i in range(d):
                models[i] = nn.Linear(d_, 1)

            start_time = time.time()

            # Create the imputer
            lin_rr_imputer = RRimputer(models, eps=epsilon, lr=lr)
            lin_imp, lin_maes, lin_rmses = lin_rr_imputer.fit_transform(
                train_miss, verbose=True, X_true=X_train_true
            )

            lin_imp = pd.DataFrame(lin_imp.detach().numpy(), columns=train_full.columns)
            lin_imp[con_col] = scaler.inverse_transform(lin_imp[con_col])

            # decode categorical
            for col in lin_imp.columns:
                if col.startswith("cat_"):
                    tem_encoder = encoders[col]
                    cate_num = len(tem_encoder.classes_)
                    lin_imp[col] = round_int(lin_imp[col], cate_num - 1)
                    lin_imp[col] = tem_encoder.inverse_transform(lin_imp[col])
                elif col.startswith("con_TS_ON"):
                    lin_imp[col] = round_to_nearest_half(lin_imp[col])
                else:
                    lin_imp[col] = round(lin_imp[col])

            # imputation time
            end_time = time.time()
            imputation_time = end_time - start_time
            imputation_times.append(imputation_time)

            lin_imp.to_csv(save_train_path, index=False)

            ################ TEST #######################
            lin_imp_test = lin_rr_imputer.transform(
                test_miss, torch.from_numpy((test_mask.astype(bool)).to_numpy())
            )

            lin_imp_test = pd.DataFrame(
                lin_imp_test.detach().numpy(), columns=train_full.columns
            )
            lin_imp_test[con_col] = scaler.inverse_transform(lin_imp_test[con_col])

            # decode categorical
            for col in lin_imp_test.columns:
                if col.startswith("cat_"):
                    tem_encoder = encoders[col]
                    cate_num = len(tem_encoder.classes_)
                    lin_imp_test[col] = round_int(lin_imp_test[col], cate_num - 1)
                    lin_imp_test[col] = tem_encoder.inverse_transform(lin_imp_test[col])
                elif col.startswith("con_TS_ON"):
                    lin_imp_test[col] = round_to_nearest_half(lin_imp_test[col])
                else:
                    lin_imp_test[col] = round(lin_imp_test[col])

            lin_imp_test.to_csv(save_test_path, index=False)

            # Compute average time
            avg_time = sum(imputation_times) / len(imputation_times)

            # Store result
            time_records.append([cohort, "MOT", method, ratio, index_file, avg_time])

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
