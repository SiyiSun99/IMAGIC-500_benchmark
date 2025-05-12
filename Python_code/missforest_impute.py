#!/home/siyi.sun/miniconda3/bin python3
# -*- coding: UTF-8 -*-
"""
@Description :   Missforest imputation
@Author      :   siyi.sun
@Time        :   2025/02/19 03:02:14
"""

import pandas as pd
import time
from pathlib import Path
from missforest import MissForest
import numpy as np
from tqdm import tqdm


def round_imputed_values(df):
    """
    Rounds imputed values in numerical columns (starting with 'con_') to 1 digit.
    """
    for col in df.columns:
        if col.startswith("con_"):  # Only process numerical columns with prefix 'con_'
            df[col] = np.round(df[col], 1)

    return df


# Define cohorts, missing methods, and ratios
cohorts = ["SynthSurvey"]



missing_methods = ["MCAR", "MAR", "MNAR"]
missing_ratios = [10, 20, 30, 40, 50]
SampleTime = 5
## NEED TO CHANGE THE PATH BASED ON YOUR COMPUTER ##
base_path = Path("/Users/siysun/Desktop/NeurIPS25/data_stored")

## NEED TO CHANGE THE PATH BASED ON YOUR COMPUTER ##
# Output path for time recording
output_file = Path(
 f"/Users/siysun/Desktop/NeurIPS25/imputation_times_mean_mode_{cohorts[0]}.csv"
)
time_records = []

for cohort in cohorts:
    full_path = base_path / f"Completed_data/{cohort}/{cohort}_train.csv"
    full_data = pd.read_csv(full_path, header=0)
    full_path_test = base_path / f"Completed_data/{cohort}/{cohort}_test.csv"
    full_data_test = pd.read_csv(full_path_test, header=0)

    categorical = [col for col in full_data.columns if col.startswith("cat_")]

    for miss_method in tqdm(missing_methods):
        for miss_ratio in tqdm(missing_ratios):
            save_dir = f"data_missforest/{cohort}/{cohort}_train/{miss_method}/miss{miss_ratio}"
            output_dir = base_path / save_dir
            output_dir.mkdir(parents=True, exist_ok=True)

            save_dir_test = (
                f"data_missforest/{cohort}/{cohort}_test/{miss_method}/miss{miss_ratio}"
            )
            output_dir_test = base_path / save_dir_test
            output_dir_test.mkdir(parents=True, exist_ok=True)

            imputation_times = []  # Store times for averaging

            for index_file in tqdm(range(SampleTime)):
                mask_path = (
                    base_path
                    / f"data_miss_mask/{cohort}/{cohort}_train/{miss_method}/miss{miss_ratio}/{index_file}.csv"
                )
                mask_n = pd.read_csv(mask_path, header=0).astype(bool)
                miss_data = full_data.mask(mask_n)

                mask_path_test = (
                    base_path
                    / f"data_miss_mask/{cohort}/{cohort}_test/{miss_method}/miss{miss_ratio}/{index_file}.csv"
                )
                mask_n_test = pd.read_csv(mask_path_test, header=0).astype(bool)
                miss_data_test = full_data_test.mask(mask_n_test)

                # Measure imputation time
                start_time = time.time()

                mf = MissForest(categorical=categorical, max_iter=20)
                imputed_data = mf.fit_transform(x=miss_data)

                end_time = time.time()
                imputation_time = end_time - start_time
                imputation_times.append(imputation_time)

                # Save imputed data
                save_path = output_dir / f"{index_file}.csv"
                imputed_data = round_imputed_values(imputed_data.mask(~mask_n))
                imputed_data.to_csv(save_path, index=False)

                imputed_data_test = mf.transform(x=miss_data_test)
                # Save imputed data
                save_path_test = output_dir_test / f"{index_file}.csv"
                imputed_data_test = round_imputed_values(imputed_data_test)
                imputed_data_test.to_csv(save_path_test, index=False)

                # Compute average time
                avg_time = sum(imputation_times) / SampleTime

                # Store result
                time_records.append(
                    [cohort, miss_method, miss_ratio, index_file, avg_time]
                )

                # Save results to CSV
                time_df = pd.DataFrame(
                    time_records,
                    columns=[
                        "Dataset",
                        "Mechanism",
                        "MissingRatio",
                        "Index_file",
                        "AvgTime",
                    ],
                )
                time_df.to_csv(output_file, index=False)
                print(f"{miss_method}, {miss_ratio} Imputation times recorded.")
