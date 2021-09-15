import os, sys, math, gc, time

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras


BASE_PATH = os.path.dirname(os.path.realpath(__file__))
MODULES_PATH = os.path.join(BASE_PATH, "..")
TF_PATH = os.path.join(BASE_PATH, "..", "..", "tf_al")
sys.path.append(MODULES_PATH)
sys.path.append(TF_PATH)

from tf_al import Config, Pool
from tf_al.wrapper import McDropout
from tf_al.utils import setup_logger
from models import fchollet_cnn, setup_growth



def log_mem_usage(logger):
    import nvidia_smi
    import os
    import psutil
    from psutil._common import bytes2human

    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    virtual_mem = psutil.virtual_memory()
    logger.info("(total) RAM: {}".format(bytes2human(virtual_mem.total)))
    logger.info("(free) RAM: {}".format(bytes2human(virtual_mem.free)))
    logger.info("(used) RAM: {}".format(bytes2human(virtual_mem.used)))

    logger.info("GPU----------")
    logger.info("Total GPU: {}".format(bytes2human(info.total)))
    logger.info("Free GPU: {}".format(bytes2human(info.free)))
    logger.info("Used GPU: {}".format(bytes2human(info.used)))
    nvidia_smi.nvmlShutdown()


# Dataset
val_set_size = 100
test_set_size = 10_000
initial_pool_size = 20
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
inputs = np.expand_dims(np.vstack([x_train, x_test])/255., axis=-1)
targets = np.hstack([y_train, y_test])
x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=test_set_size)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_set_size)
pool = Pool(x_train, y_train)
pool.init(10)

# Model
setup_growth()

sample_size = 25
batch_size = 900
num_classes = len(np.unique(y_train))
base_model = fchollet_cnn(output=num_classes)

fit_params = {"epochs": 200, "batch_size": batch_size}
mc_config = Config(
    fit=fit_params,
    query={"sample_size": sample_size},
    eval={"batch_size": 900, "sample_size": sample_size}
)

mc_model = McDropout(base_model, config=mc_config)

optimizer = "adam"
loss = "sparse_categorical_crossentropy"
metrics = [keras.metrics.SparseCategoricalAccuracy()]
mc_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


def reset_step(self, pool, dataset):
    """
        Overwrite reset function after each acquisiton iteration.

        Parameters:
            pool (Pool): Pool of labeled datapoints.
            dataset (Dataset): dataset object containing train, test and eval sets.
    """
    self._model = fchollet_cnn(output=num_classes)
    self._model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

setattr(McDropout, "reset", reset_step)

# Loop
step_size = 500
i = 0
logger = setup_logger(True, "Debug loop")
while True:

    logger.info("---- Iteration {}".format(i))
    
    data, indices = pool.get_unlabeled_data()
    num_datapoints = len(data)
    num_batches = math.ceil(num_datapoints/batch_size)
    batches = np.array_split(data, num_batches, axis=0)

    start = time.time()
    logger.info("Num Datapoints: {}".format(num_datapoints))
    max_entropy = mc_model.get_query_fn("max_entropy")
    result = []
    for batch in batches:
        sub_result = max_entropy(tf.convert_to_tensor(batch), sample_size=sample_size)
        result.append(sub_result))

    i+=1
    np.hstack(result)
    indices = pool.get_unlabeled_indices()
    selected = np.random.choice(indices, step_size, replace=False)
    pool.annotate(selected)
    logger.info("Query-Time: {}".format(time.time() - start))

    gc.collect()
    tf.keras.backend.clear_session()

    log_mem_usage(logger)
    logger.info("-------------------------------")
