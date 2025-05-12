import pandas as pd
import numpy as np
from pathlib import Path
import time
from hyperimpute.plugins.imputers import Imputers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from hyperimpute.utils.benchmarks import compare_models
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

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

cohorts = ["SynthSurvey"]
base_path = Path("/Users/siysun/Desktop/NeurIPS25/data_stored")
methods = ["MCAR", "MAR", "MNAR"]
ratios = [50, 40, 30, 20, 10]
Sampletime = 5
impute_method = "hyperimpute"

## NEED TO CHANGE THE PATH BASED ON YOUR COMPUTER ##
# Output path for time recording
output_file = Path(
 f"/Users/siysun/Desktop/NeurIPS25/imputation_times_{impute_method}_{cohorts[0]}.csv"
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
    if col.startswith("cat_") and col != "cat_hid":
        tem_encoder = LabelEncoder()
        train_full[col] = tem_encoder.fit_transform(train_full[col])
        test_full[col] = tem_encoder.transform(test_full[col])
        encoders[col] = tem_encoder

for method in tqdm(methods):
    for ratio in tqdm(ratios):
        save_dir_train = (
            base_path
            / f"data_{impute_method}/{cohort}/{cohort}_train/{method}/miss{ratio}"
        )
        save_dir_test = (
            base_path
            / f"data_{impute_method}/{cohort}/{cohort}_test/{method}/miss{ratio}"
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
            train_miss = train_full.mask(train_mask.astype(bool))
            test_miss = test_full.mask(test_mask.astype(bool))

            plugin = Imputers().get(
                impute_method,  # the name of the imputation method.
                # The rest of the kwargs are specific to the method
                optimizer="hyperband",
                classifier_seed=["logistic_regression"],
                regression_seed=["linear_regression"],
                class_threshold=2,
            )

            start_time = time.time()

            ## Impute train
            out_train = plugin.fit_transform(train_miss.copy())
            out_train.columns = train_miss.columns
            out_train[con_col] = scaler.inverse_transform(out_train[con_col])

            # decode categorical
            for col in out_train.columns:
                if col == "cat_hid":
                    pass
                elif col.startswith("cat_"):
                    tem_encoder = encoders[col]
                    cate_num = len(tem_encoder.classes_)
                    out_train[col] = round_int(out_train[col], cate_num - 1)
                    out_train[col] = tem_encoder.inverse_transform(out_train[col])
                else:
                    out_train[col] = round(out_train[col])

            # imputation time
            end_time = time.time()
            imputation_time = end_time - start_time
            imputation_times.append(imputation_time)

            out_train.to_csv(save_train_path, index=False)

            ## Impute test
            out_test = plugin.transform(test_miss.copy())
            out_test.columns = test_miss.columns
            out_test[con_col] = scaler.inverse_transform(out_test[con_col])

            # decode categorical
            for col in out_test.columns:
                if col == "cat_hid":
                    pass
                elif col.startswith("cat_"):
                    tem_encoder = encoders[col]
                    out_test[col] = round_int(out_test[col], cate_num - 1)
                    out_test[col] = tem_encoder.inverse_transform(out_test[col])
                else:
                    out_test[col] = round(out_test[col])

            out_test.to_csv(save_test_path, index=False)

            # Compute average time
            avg_time = sum(imputation_times) / len(imputation_times)

            # Store result
            time_records.append(
                [cohort, impute_method, method, ratio, index_file, avg_time]
            )

            # Save results to CSV
            time_df = pd.DataFrame(
                time_records,
                columns=[
                    "Dataset",
                    "Method",
                    "Mechanism",
                    "MissingRatio",
                    "Index_file",
                    "AvgTime",
                ],
            )
            time_df.to_csv(output_file, index=False)
            print(f"{method}, {ratio} Imputation times recorded.")
