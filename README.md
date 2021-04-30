
# AI Workspace Template

A simple workspace template for deep-/machine-learning projects using python/anaconda, jupyter and docker.


[How to win data-science competition, learn from top kagglers](https://www.coursera.org/learn/competitive-data-science)

[Lin Regression Pytorch](https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817)

[Probability concepts explained](https://towardsdatascience.com/probability-concepts-explained-maximum-likelihood-estimation-c7b4342fdbb1#:~:text=Maximum%20likelihood%20estimation%20is%20a%20method%20that%20will%20find%20the,that%20best%20fits%20the%20data.&text=The%20goal%20of%20maximum%20likelihood,probability%20of%20observing%20the%20data.)


# Index

1. [Prerequisite](#Prerequisite)
2. [Setup](#Setup)
3. [Datasets](#Datasets)
3. [Contribution](#Contribution)



## Prerequisite

The following dependencies need to be installed to 

- [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/)
- (optional) Anaconda/Python 3.7




## Setup

1. Clone this repository
 
```
git clone https://github.com/ExLeonem/ai-workspace-template
```

2. Run the setup scripts
3. Configure the workspace depending your needs.


### Multi Cuda Setup

- cuda-toolkit-11.0 (for tensorflow 2.4.1)
- cuda-toolkit-10.2 (for pytorch)

```
    $ sudo apt-get install cuda-toolkit-11-0
```

sudo apt-key adv --refresh-keys --keyserver keyserver.ubuntu.com

https://medium.com/@peterjussi/multicuda-multiple-versions-of-cuda-on-one-machine-4b6ccda6faae


### Performing Moment-Propagation

Create new environment Anaconda environment with python version 3.8

```
    $ conda create -n <environment_name> python==3.7.0 ipython
```


Install pip for anaconda 

```
    $ conda install pip
```



Install packages of `requirements.txt` 

## Read datasets from kaggle

Create a kaggle.json file inside of the config directory.

```json
{
    "username":"user_name",
    "key":"key_from_kaggle"
}
```




## Datasets

### Regression

1. [Insurance Forecast](https://www.kaggle.com/mirichoi0218/insurance/discussion)



## Resources

### Encoding

[Categorical Data Encoding](https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02)


## Contribution

To contribute to this repository open a new issue to discuss the changes to be made. 
Afterwards fork this repository and apply the changes. When creating a pull request, make sure
to link the issue.