

# TODO


https://github.com/tensorflow/tensorflow/issues/31842


- [ ] MomentPropagation get's killed off by OOM Process Killer
    - [ ] Try eager mode
    - [ ] Check if model builds multiple different graphs
    - [ ] Try custom fitting loop?

- [ ] Pool initialization and annotation using different targets (regression, one-hot vectors, ...)


# How to add a new model type


# Index

1. [Guide](#Guide)
    1. [Model Wrapper](#Model-Wrapper)
    2. [Custom Model Wrapper](#Custom-Model-Wrapper)
    3. [Generic Active Learning loop](#Generic-Active-Learning-loop)
    4. [Custom Active Learning loop](#Custom-Active-Learning-loop)
    2. [Experiment Suit](#Experiment-Suit)
2. [Scripts](#Scripts)


# Guide

The active learning library consists of an high level API which can be used
to create generic experiments, as well as low level components that can be 
used to create custom active learning loops.


## Model Wrapper


## Custom Model Wrapper


## Generic Active Learning loop

To create a high level learning loop use the `ActiveLearnigLoop` class.

```python

from active_learning import AcquistionFunction, Dataset, ActiveLearningLoop
from bayesian import McDropout
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Softmax

# Create and wrapp the model
base_model = Sequential([
        Conv2D(128, 4,activation="relu", input_shape=(28, 28, 1)),
        MaxPool2D(),
        Dropout(.2),
        Conv2D(64, 3, activation="relu"),
        MaxPool2D(),
        Dropout(.2),
        Flatten(),
        Dense(512, activation="relu"),
        Dense(256, activation="relu"),
        Dense(128, activation="relu"),
        Dense(5, activation="softmax")
    ])
mc_model = McDropout(base_model)

# Create a pool for unlabeled and labeled data
inputs, targets = load_mnist_data()
pooled_dataset = Dataset(inputs, targets, init_size=1)

# Create an iterable loop object
query_fn = "max_entropy"
al_iterations = ActiveLearningIterator(mc_model, pooled_dataset, query_fn)

for meta in al_iterations:
    # meta returns information of i-th iteration
    # Accumulated information can now be used to be written out or else
    pass
```


## Custom Active Learning loop


## Experiments



# Scripts

## Create documentation

To create documentation for the `./modules` directory. Execute following command
in `./docs`

```shell
$ make html
```

## Run tests

To perform automated unittests run following command in the workspace directory `./wp`.

```shell
$ pytest
```

To generate additional coverage reports run.

```shell
$ pytest --cov
```



## Clear log files

Clear all generated logs in the `./modules/log` directory.

```shell
$ python clear.py
```

