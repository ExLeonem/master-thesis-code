
# Accelerating BNNs in the context of Active Learning




# Index

1. [Setup](#Setup)
2. [Directories](#Directories)
3. [Scripts](#Scripts)

## Setup


1. Clone this repository
```shell
git clone https://github.com/ExLeonem/master-thesis-code
```

2. Run the setup script to create non-existent directories.

```shell
python ./config/setup.py
```

3. Create a conda environment using the environment.yml in `./config`
```shell
conda env import ./config/base_env.yml
```


### Manual Environment Setup

If the automated creation of the environment fails following packages are needed to be installed manually to get a working workspace.

| Package | Tested version
| --- | ---
| tensorflow | 2.2.0
| tensorflow-gpu | 2.2.0
| tensorflow-probability | 0.7
| tf-al | 0.1.1
| tf-al-mp | master-branch
| numpy | 1.18.5
| tqdm | 4.59.0
| pytest | 6.2.4
| pytest-cov | 2.12.1
| sphinx | 3.2.1


</br>
Additional packages to be installed when executing jupyter notebooks:

| Package | Tested version
| --- | ---
| pandas | 0.25.3
| matplotlib | 3.3.4
| seaborn | 0.11.1


## Directories

| Directory | Description
| --- | ---
| `wp/metrics` | Contains all metrics acquired from executing the experiments
| `wp/modules` | Contains experiment scripts
| `wp/notebooks` | Jupyter notebooks


### Metrics Directory


## Experiments

All experiments are located in `./wp/modules/experiments`.


## Scripts


### Jupyter notebooks

Move into the folder `./wp/notebooks` and execute

```shell
jupyter notebook
```