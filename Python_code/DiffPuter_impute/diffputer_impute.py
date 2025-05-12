import os
import torch
import numpy as np
import pandas as pd
import json
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
import time
from pathlib import Path
from tqdm import tqdm

from diffPuter_model import MLPDiffusion, Model
from diffPuter_dataset import load_dataset, recover_num_cat
from diffusion_utils import sample_step, impute_mask, round_to_nearest_half, round_int

warnings.filterwarnings("ignore")

## NEED TO CHANGE THE PATH BASED ON YOUR COMPUTER ##
base_path = Path("/Users/siysun/Desktop/PhD/SynthCPHS_benchmark/data_stored")
## NEED TO CHANGE THE PATH BASED ON YOUR COMPUTER ##
# Output path for time recording
output_file = Path(
    f"/Users/siysun/Desktop/PhD/SynthCPHS_benchmark/imputation_times_diffputer_C19_Synthetic.csv"
)
time_records = []

# Parameters
max_iter = 4  # 4 from paper figure 3
num_trials = 10  # 10 from figure 4
num_steps = 20  # 50 from figure 5
hid_dim = 1024

# check cuda
if torch.cuda.is_available():
    device = f"cuda"
else:
    device = "cpu"

cohorts = ["C19"]
miss_methods = ["MCAR"]
miss_ratios = [10, 20, 30, 40, 50]
Sampletime = 5

for cohort in cohorts:
    data_path = base_path / f"Completed_data/{cohort}/{cohort}_test.csv"
    data_test_df = pd.read_csv(data_path)
    all_cols = data_test_df.columns
    con_col = [col for col in data_test_df.columns if col.startswith("con")]
    cat_col = [col for col in data_test_df.columns if col.startswith("cat")]
    for miss_method in tqdm(miss_methods):
        for miss_ratio in tqdm(miss_ratios):
            save_dir_train = (
                base_path
                / f"data_diffputer/{cohort}/{cohort}_train/{miss_method}/miss{miss_ratio}"
            )
            save_dir_test = (
                base_path
                / f"data_diffputer/{cohort}/{cohort}_test/{miss_method}/miss{miss_ratio}"
            )
            save_dir_train.mkdir(parents=True, exist_ok=True)
            save_dir_test.mkdir(parents=True, exist_ok=True)

            imputation_times = []  # Store times for averaging

            for index_file in tqdm(range(Sampletime)):
                # path for save imputed dataset
                save_train_path = save_dir_train / f"{index_file}.csv"
                save_test_path = save_dir_test / f"{index_file}.csv"

                (
                    train_X,
                    test_X,
                    ori_train_mask,
                    ori_test_mask,
                    train_num,
                    test_num,
                    train_cat_idx,
                    test_cat_idx,
                    train_mask,
                    test_mask,
                    cat_bin_num,
                ) = load_dataset(base_path, cohort, miss_method, miss_ratio, index_file)

                mean_X = train_X.mean(0)
                std_X = train_X.std(0)
                in_dim = train_X.shape[1]

                X = (train_X - mean_X) / std_X / 2
                X = torch.tensor(X)

                X_test = (test_X - mean_X) / std_X / 2
                X_test = torch.tensor(X_test)

                mask_train = torch.tensor(train_mask)
                mask_test = torch.tensor(test_mask)

                total_time = 0

                for iteration in range(max_iter):

                    ## M-Step: Density Estimation
                    ckpt_dir = (
                        base_path
                        / f"ckpt/{cohort}/rate{miss_ratio}/{miss_method}/{num_trials}_{num_steps}"
                    )
                    ckpt_ite = ckpt_dir / f"{iteration}"
                    ckpt_ite.mkdir(parents=True, exist_ok=True)

                    if iteration == 0:
                        X_miss = (1.0 - mask_train.float()) * X
                        train_data = X_miss.numpy()
                    else:
                        print(f"Loading X_miss from {ckpt_dir}/iter_{iteration}.npy")
                        X_miss = np.load(f"{ckpt_dir}/iter_{iteration}.npy") / 2
                        train_data = X_miss

                    batch_size = 4096
                    train_loader = DataLoader(
                        train_data,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=4,
                    )

                    num_epochs = 200 + 1

                    denoise_fn = MLPDiffusion(in_dim, hid_dim).to(device)

                    num_params = sum(p.numel() for p in denoise_fn.parameters())

                    model = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)

                    optimizer = torch.optim.Adam(
                        model.parameters(), lr=1e-4, weight_decay=0
                    )
                    scheduler = ReduceLROnPlateau(
                        optimizer, mode="min", factor=0.9, patience=40, verbose=True
                    )

                    model.train()

                    best_loss = float("inf")
                    patience = 0
                    start_time = time.time()
                    for epoch in tqdm(range(num_epochs)):

                        batch_loss = 0.0
                        len_input = 0
                        for batch in train_loader:
                            inputs = batch.float().to(device)
                            loss = model(inputs)

                            loss = loss.mean()
                            batch_loss += loss.item() * len(inputs)
                            len_input += len(inputs)

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                        curr_loss = batch_loss / len_input
                        scheduler.step(curr_loss)

                        if curr_loss < best_loss:
                            best_loss = curr_loss
                            patience = 0
                            torch.save(
                                model.state_dict(), f"{ckpt_dir}/{iteration}/model.pt"
                            )
                        else:
                            patience += 1
                            if patience == 200:
                                print("Early stopping")
                                break
                        print(
                            f"Epoch {epoch+1}/{num_epochs}, Loss: {curr_loss:.4f}, Best loss: {best_loss:.4f}"
                        )

                        if epoch % 1000 == 0:
                            torch.save(
                                model.state_dict(),
                                f"{ckpt_dir}/{iteration}/model_{epoch}.pt",
                            )

                    end_time = time.time()
                    total_time += end_time - start_time

                    # In-sample imputation

                    rec_Xs = []

                    for trial in range(num_trials):

                        X_miss = (1.0 - mask_train.float()) * X
                        X_miss = X_miss.to(device)
                        impute_X = X_miss

                        in_dim = X.shape[1]

                        denoise_fn = MLPDiffusion(in_dim, hid_dim).to(device)

                        model = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)
                        model.load_state_dict(
                            torch.load(f"{ckpt_dir}/{iteration}/model.pt")
                        )

                        # ==========================================================

                        net = model.denoise_fn_D

                        num_samples, dim = X.shape[0], X.shape[1]
                        rec_X = impute_mask(
                            net,
                            impute_X,
                            mask_train,
                            num_samples,
                            dim,
                            num_steps,
                            device,
                        )

                        mask_int = mask_train.to(torch.float).to(device)
                        rec_X = rec_X * mask_int + impute_X * (1 - mask_int)
                        rec_Xs.append(rec_X)

                        print(f"Trial = {trial}")

                    rec_X = torch.stack(rec_Xs, dim=0).mean(0)

                    rec_X = rec_X.cpu().numpy() * 2
                    X_true = X.cpu().numpy() * 2

                    np.save(f"{ckpt_dir}/iter_{iteration+1}.npy", rec_X)

                pred_X = rec_X[:]
                len_num = train_num.shape[1]

                res = pred_X[:, len_num:] * std_X[len_num:] + mean_X[len_num:]
                pred_X[:, len_num:] = res

                pred_num = pred_X[:, :len_num] * std_X[:len_num] + mean_X[:len_num]
                imputed_con = pd.DataFrame(pred_num, columns=con_col)

                pred_cat = recover_num_cat(
                    pred_X[:, len_num:], len(cat_bin_num), cat_bin_num
                )
                imputed_cat = pd.DataFrame(pred_cat, columns=cat_col)

                for col in imputed_cat.columns:
                    map_path_cat = (
                        base_path / f"Completed_data/{cohort}/{col}_map_cat.json"
                    )
                    with open(map_path_cat, "r") as f:
                        idx_to_category = json.load(f)

                    imputed_cat[col] = imputed_cat[col].apply(
                        lambda x: idx_to_category[str(x)]
                    )

                imputed_data = imputed_con.join(imputed_cat)[all_cols]
                # decode categorical
                for col in imputed_data.columns:
                    if col.startswith("con_TS_ON"):
                        imputed_data[col] = round_to_nearest_half(imputed_data[col])
                    elif col.startswith("con_"):
                        imputed_data[col] = round(imputed_data[col])
                    else:
                        pass

                imputed_data.to_csv(save_train_path, index=False)

                ####################### TEST ################3

                rec_Xs = []

                for trial in range(num_trials):

                    # For out-of-sample imputation, no results from previous iterations are used

                    X_miss = (1.0 - mask_test.float()) * X_test
                    X_miss = X_miss.to(device)
                    impute_X = X_miss

                    in_dim = X_test.shape[1]

                    denoise_fn = MLPDiffusion(in_dim, hid_dim).to(device)

                    model = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)
                    model.load_state_dict(
                        torch.load(f"{ckpt_dir}/{iteration}/model.pt")
                    )

                    # ==========================================================
                    net = model.denoise_fn_D

                    num_samples, dim = X_test.shape[0], X_test.shape[1]
                    rec_X = impute_mask(
                        net, impute_X, mask_test, num_samples, dim, num_steps, device
                    )

                    mask_int = mask_test.to(torch.float).to(device)
                    rec_X = rec_X * mask_int + impute_X * (1 - mask_int)
                    rec_Xs.append(rec_X)

                    print(f"Trial = {trial}")

                rec_X = torch.stack(rec_Xs, dim=0).mean(0)

                rec_X = rec_X.cpu().numpy() * 2
                X_true = X.cpu().numpy() * 2

                pred_X = rec_X[:]
                len_num = train_num.shape[1]

                res = pred_X[:, len_num:] * std_X[len_num:] + mean_X[len_num:]
                pred_X[:, len_num:] = res

                pred_num = pred_X[:, :len_num] * std_X[:len_num] + mean_X[:len_num]
                imputed_con = pd.DataFrame(pred_num, columns=con_col)

                pred_cat = recover_num_cat(
                    pred_X[:, len_num:], len(cat_bin_num), cat_bin_num
                )
                imputed_cat = pd.DataFrame(pred_cat, columns=cat_col)

                for col in imputed_cat.columns:
                    map_path_cat = (
                        base_path / f"Completed_data/{cohort}/{col}_map_cat.json"
                    )
                    with open(map_path_cat, "r") as f:
                        idx_to_category = json.load(f)

                    imputed_cat[col] = imputed_cat[col].apply(
                        lambda x: idx_to_category[str(x)]
                    )

                imputed_data = imputed_con.join(imputed_cat)[all_cols]
                # decode categorical
                for col in imputed_data.columns:
                    if col.startswith("con_TS_ON"):
                        imputed_data[col] = round_to_nearest_half(imputed_data[col])
                    elif col.startswith("con_"):
                        imputed_data[col] = round(imputed_data[col])
                    else:
                        pass

                imputed_data.to_csv(save_test_path, index=False)

                imputation_times.append(total_time)
                # Compute average time
                avg_time = sum(imputation_times) / len(imputation_times)

                # Store result
                time_records.append(
                    [cohort, "diffputer", miss_method, miss_ratio, index_file, avg_time]
                )

                # Save results to CSV
                time_df = pd.DataFrame(
                    time_records,
                    columns=[
                        "Dataset",
                        "Method",
                        "Mechanism",
                        "MissingRatio",
                        "Index File",
                        "AvgTime",
                    ],
                )
                time_df.to_csv(output_file, index=False)
                print(f"{miss_method}, {miss_ratio} Imputation times recorded.")
