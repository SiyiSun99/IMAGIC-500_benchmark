# 📊 IMAGIC-500: IMputation benchmark on A Generative Imaginary Country (500k samples)

<p align="center">
  <img src="https://img.shields.io/badge/Dataset-IMAGIC--500-blue" alt="Dataset Badge">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License Badge">
  <img src="https://img.shields.io/badge/Benchmark-14%20Methods-orange" alt="Benchmark Badge">
  <img src="https://img.shields.io/badge/Missingness-MCAR%2FMAR%2FMNAR-purple" alt="Missingness Badge">
  <img src="https://img.shields.io/badge/Missing%20Ratio-10%25--50%25-darkred" alt="Missing Ratio Badge">
</p>

Welcome to the official repository for **IMAGIC-500**, a large-scale synthetic benchmark dataset for evaluating missing data imputation methods in structured socioeconomic surveys introduced in our NeurIPS 2025 paper:

> 📝 *"IMAGIC-500: IMputation benchmark on A Generative Imaginary Country (500k samples)"*

Derived from the **World Bank's public Synthetic Data for an Imaginary Country (SDIC)** dataset, IMAGIC-500 contains 500,000 individuals across 100,000 households, capturing 19 key socioeconomic variables. This repository provides both the dataset and a comprehensive benchmark framework for testing imputation techniques under different missingness mechanisms (MCAR, MAR, MNAR) and ratios. Whether you're working with classical statistics, machine learning, or deep generative models, IMAGIC-500 is designed to support **robust, reproducible, and fair evaluation**.



---

## 🌟 Key Features
- 📈 **500k samples** with **19 socio-economic variables** (demographics, education, occupation, household expenditures, assets, etc)
- 🗂️ Nested structure: individual → household → district → province
- 🔍 Simulated missingness under **MCAR**, **MAR**, and **MNAR** mechanisms at varying ratios (10%–50%)
- 🛠️ Comprehensive benchmark of **14 imputation methods**, covering statistical, machine learning, and deep learning approaches
- ⚙️ Evaluation framework for:
  - **Imputation accuracy** (RMSE, F1)
  - **Downstream task performance** (Weighted ROC-AUC degradation)
  - **Computational efficiency** (runtime)

---

## 🚀 Quick Start

### Step 1: Clone the Repository

First, clone the repository and navigate into the directory:

```bash
git clone https://github.com/yourusername/IMAGIC-500_benchmark.git
cd IMAGIC-500_benchmark
```

### Step 2: Download the Dataset

Download the synthetic dataset (`SynthSurvey.csv`) from Harvard Dataverse [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/7GMPBH) and save it to the following path within the repository:

```
data_stored/Completed_data/SynthSurvey/SynthSurvey_all.csv
```

---

## 📁 Folder Structure

```
.
├── Evaluation                  # Scripts for performance evaluation
│   ├── TabCSDI_post_processing.py
│   ├── Downstream_performance.py
│   └── Imputation_performance.py
├── Generate_Missingness        # R scripts to generate missing data
│   ├── amputation.R
│   └── sample_miss.R
├── Preprocessing               # Data preprocessing utilities
│   ├── map_missing_mask.py
│   └── train_test_split.py
├── Python_code                 # Implementation of imputation methods
│   ├── DiffPuter_impute/
│   ├── GAIN_impute.py
│   ├── MICE_impute.py
│   ├── MOT_impute/
│   ├── Remasker_impute/
│   ├── TabCSDI_impute/
│   ├── dsan_impute.py
│   ├── dsn_impute.py
│   ├── hyperimpute_impute.py
│   ├── mean_mode_impute.py
│   ├── miracle_impute.py
│   ├── missforest_impute.py
│   ├── miwae_impute.py
│   └── softimpute_impute.py
├── data_stored                 # Raw and processed data files
│   └── Completed_data/
│       └── SynthSurvey/
│           └── SynthSurvey_all.csv
├── README.md
└── requirements.txt            # Required Python packages
```

---

## ⚡ Imputation methods

| Method       | Description                                                                                             | Code |
|--------------|---------------------------------------------------------------------------------------------------------|------|
| [DiffPuter](https://github.com/hengruizhang98/DiffPuter)    | Diffusion-based deep learning imputation leveraging diffusion probabilistic models to recover missing values. | [DiffPuter_impute](Python_code/DiffPuter_impute/) |
| [MOT](https://proceedings.mlr.press/v119/muzellec20a.html)          | Imputation based on the Masked Optimal Transport method, using optimal transport theory for missing data imputation. | [MOT_impute](Python_code/MOT_impute/) |
| [Remasker](https://arxiv.org/abs/2309.13793)     | Iterative masking and refining method for imputation tasks, suitable for complex missingness patterns. | [Remasker_impute](Python_code/Remasker_impute/) |
| [TabCSDI](https://github.com/pfnet-research/TabCSDI)      | Tabular Conditional Score-based Diffusion Imputation (TabCSDI), leveraging conditional diffusion models to handle tabular missing data. | [TabCSDI_impute](Python_code/TabCSDI_impute/) |
| [GAIN](https://proceedings.mlr.press/v80/yoon18a.html?ref=https://githubhelp.com)         | Generative Adversarial Imputation Nets (GAIN), using GAN-based deep learning for imputation.           | [GAIN_impute.py](Python_code/GAIN_impute.py) |
| [MICE](https://www.jstatsoft.org/article/view/v045i03)         | Multiple Imputation by Chained Equations (MICE), iterative imputation using regression models.          | [MICE_impute.py](Python_code/MICE_impute.py) |
| [DSAN](https://www.mdpi.com/2306-5729/8/6/102)         | Deep Self-Attention Networks (DSAN), attention-based neural networks enhancing feature focus during imputation. | [dsan_impute.py](Python_code/dsan_impute.py) |
| DSN          | A variation of DSAN without attention mechanisms.        | [dsn_impute.py](Python_code/dsn_impute.py) |
| [HyperImpute](https://proceedings.mlr.press/v162/jarrett22a.html)  | A generalized method integrates iterative imputation with automatic model selection. | [hyperimpute_impute.py](Python_code/hyperimpute_impute.py) |
| Mean/Mode    | Basic statistical method imputing missing values with mean (numerical) or mode (categorical) of each feature. | [mean_mode_impute.py](Python_code/mean_mode_impute.py) |
| [MIRACLE](https://proceedings.neurips.cc/paper/2021/hash/c80bcf42c220b8f5c41f85344242f1b0-Abstract.html)      | Causally-aware imputation using causal structure learning methods for data recovery.                    | [miracle_impute.py](Python_code/miracle_impute.py) |
| [MissForest](https://academic.oup.com/bioinformatics/article/28/1/112/219101?login=true)   | Random Forest-based iterative imputation, suitable for both numerical and categorical features.         | [missforest_impute.py](Python_code/missforest_impute.py) |
| [MIWAE](https://proceedings.mlr.press/v97/mattei19a.html?ref=https://githubhelp.com)        | Deep generative modelling and imputation using importance-weighted autoencoder architectures.           | [miwae_impute.py](Python_code/miwae_impute.py) |
| [SoftImpute](https://dl.acm.org/doi/abs/10.5555/2789272.2912106)   | Matrix completion using low-rank approximation via nuclear-norm regularization for large datasets.      | [softimpute_impute.py](Python_code/softimpute_impute.py) |


---

## 🛠️ Step-by-Step Guide to Reproduce Results

### 1. 🧹 Data Preprocessing

To split the dataset into training and testing sets, run:

```bash
python Preprocessing/train_test_split.py
```

- This will generate two CSV files (`SynthSurvey_train.csv` and `SynthSurvey_test.csv`) containing 18 features.
- The target variable (`cat_educ_attain`) for train and test sets is stored separately in `SynthSurvey_train_y.csv` and `SynthSurvey_test_y.csv`.

### 2. 🎲 Generate Missingness (R Script)

Generate missing data scenarios using R:

```R
Rscript Generate_Missingness/sample_miss.R
```

- This creates datasets with simulated missingness according to predefined mechanisms.
- The target variable and machine-generated variables remain fully observed.

### 3. 🗂️ Create Missing Masks

To create binary masks indicating missing data points, run:

```bash
python Preprocessing/map_missing_mask.py
```

- Masks are saved as binary indicators (`0`: not missing, `1`: missing).
- You can delete the `data_miss` folder afterward if no longer needed, as masks in `data_miss_mask` will suffice for imputation.

### 4. ⚙️ Perform Imputation

Run imputation methods individually. For example, for Mean/Mode imputation:

```bash
python Python_code/mean_mode_impute.py
```

- Ensure the `base_path` in each script matches your directory structure.
- The script outputs imputed datasets and logs runtime.
- After each run, check the logs and use the entry with the highest `index_file`, representing average time across runs.
- For TabCSDI_impute, modify the `featuredim ` in `Python_code/TabCSDI_impute/parameters/config_TabCSDI.yaml`

### 5. 📈 Evaluation

Before evaluation, if you are using TabCSDI-imputed datasets, you must first run the post-processing script to restore the `cat_hid` (household ID) variable, which is not missing but was dropped due to TabCSDI's modeling characteristics.

```bash
python Evaluation/TabCSDI_post_processing.py
```

To evaluate imputation performance:

```bash
python Evaluation/Imputation_performance.py
```

To evaluate downstream classification performance:

```bash
python Evaluation/Downstream_performance.py
```

- Runtime efficiency can be directly assessed via the runtime CSV files generated during imputation.

---

## 💻 Computational Resources Used
- **Operating System**: Rocky Linux 8 (based on Red Hat Enterprise Linux 8)
- **CPU**: 2x AMD EPYC 7763 64-Core Processor 1.8GHz (128 cores in total)
- **RAM**: 1000 GiB
- **GPU**: 4x NVIDIA A100-SXM-80GB GPUs (each with 6912 FP32 CUDA cores)
- **Interconnect**: Dual-rail Mellanox HDR200 InfiniBand
- **Cluster**: 90 Dell PowerEdge XE8545 servers
- **Software**: CUDA 11.4, Python 3.9.20, PyTorch 2.6.0

---

## 📦 Dependencies

### R Packages

Install the required R packages:

- `mice`, `parallel`, `gdata`

### Python Packages

Install all required Python packages using:

```bash
pip install -r requirements.txt
```

---

## 📜 License

This project is licensed under the **MIT License**.

---

✨ **Star this repository if you find it useful!** ✨
