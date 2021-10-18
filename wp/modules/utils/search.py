import os,sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from hyperopt import tpe, fmin, hp

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
MODULE_PATH = os.path.join(BASE_PATH, "..")
sys.path.append(MODULE_PATH)
from models import dnn_dense_dropout

TF_PATH = os.path.join(BASE_PATH, "..", "..", "tf_al")
sys.path.append(TF_PATH)
from tf_al.utils import setup_logger



def smbo(
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    parameter,
    trials=5
):
    history = {}
    for t in range(trials):
        pass

    return history



def search_initial_pool_size(
    model, 
    x_train, 
    y_train,
    x_test,
    y_test,
    sizes=None, 
    threshold=0.5, 
    max_acc=None, 
    epochs=100, 
    batch_size=10,
    verbose=False
):
    """

    """
    logger = setup_logger(verbose, "Parameter search")

    if sizes is None:
        sizes = [10, 30, 60, 120, 240, 480, 960]

    accuracies = []
    indices = np.arange(0, len(x_train))
    for idx in range(len(sizes)):
        size = sizes[idx]
        selected = np.random.choice(indices, size, replace=False)
        x_train_sub = x_train[selected]
        y_train_sub = y_train[selected]

        model.fit(x_train_sub, y_train_sub, epochs=epochs, batch_size=batch_size, verbose=0)
        loss, acc = model.evaluate(x_test, y_test, verbose=0)

        logger.info("Size: {} ~ Accuracy: {}".format(size, acc))
        accuracies.append(acc)
        if len(accuracies) > 1 and \
            (accuracies[idx] - accuracies[idx-1]) < threshold:
            logger.info("Difference below threshold: {}".format(accuracies[idx] - accuracies[idx-1]))
            return sizes[idx-1]
        
        if max_acc is not None and acc >= max_acc:
            return sizes[idx]

        keras.backend.clear_session()
        

    return sizes[-1]


def objective(hyperparameters):
    pass