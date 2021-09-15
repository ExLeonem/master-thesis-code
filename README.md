
# AI Workspace Template

A simple workspace template for deep-/machine-learning projects using python/anaconda, jupyter and docker.


[How to win data-science competition, learn from top kagglers](https://www.coursera.org/learn/competitive-data-science)

[Lin Regression Pytorch](https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817)

[Probability concepts explained](https://towardsdatascience.com/probability-concepts-explained-maximum-likelihood-estimation-c7b4342fdbb1#:~:text=Maximum%20likelihood%20estimation%20is%20a%20method%20that%20will%20find%20the,that%20best%20fits%20the%20data.&text=The%20goal%20of%20maximum%20likelihood,probability%20of%20observing%20the%20data.)


# Create Gaussian Process model and compare
[GPFlow](https://github.com/GPflow/GPflow)
https://gpflow.readthedocs.io/en/master/notebooks/intro.html#Advanced-needs


# Create Adaption for Batch Active Learning



# Index

1. [Prerequisite](#Prerequisite)
2. [Setup](#Setup)
3. [Datasets](#Datasets)
3. [Contribution](#Contribution)


## Prerequisite

- [Docker](https://www.docker.com/) (optional) 
- [Docker Compose](https://docs.docker.com/compose/) (optional)
- [Anaconda](https://www.anaconda.com/)


## Setup

1. Clone this repository
 
```
git clone https://github.com/ExLeonem/ai-workspace-template
```

2. Run the setup script to create non-existent directories.

```shell
$ python ./config/setup.py
```


3. Configure the workspace depending your needs. (optional)

4. Import or create the conda environment. </br> To import the conda environment use one of following commands, depending your OS.

```
$ conda env import ./config/env_linux.yml 
```

```
$ conda env import ./config/env_win.yml
```



### Manual Environment Setup

If the automated creation of the environment fails following packages are needed to be installed manually to get a working workspace.

| Package | Tested version
| --- | ---
| tensorflow | 2.2.0
| tensorflow-gpu | 2.2.0
| tensorflow-probability | 0.7
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



## Datasets

Following datasets are needed to perform defined pre-defined experiments:

1. MNIST
2. FashionMNIST
3. caltech_ucsd_birds200


## Contribution

To contribute to this repository open a new issue to discuss the changes to be made. 
Afterwards fork this repository and apply the changes. When creating a pull request, make sure
to link the issue.