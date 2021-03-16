import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import tensorflow_probability as tfp


# https://keras.io/examples/keras_recipes/bayesian_neural_networks/
# Keras Bayesian Neural Network tutorial


def get_train_test_splits(train_size, batch_size=1):
    """
        We prefetch with a buffer the same size as the dataset because th dataset.
        Is very small and fits into memory
    """
    dataset = (
        tfds.load(name="titanic", as_supervised=True, split="train")
        .map(lambda x, y: (x, tf.cast(y, tf.float32)))
        .prefetch(buffer_size=dataset_size)
        .cache()
    )

    # We shuffle with a buffer the same size as the dataset
    train_dataset = (
        dataset.take(train_size).shuffle(buffer_size=train_size).batch(batch_size)
    )

    test_dataset = dataset.skip(train_size).batch(batch_size)
    return train_dataset, test_dataset



hidden_units = [8, 8]
learning_rate = .001
def run_experiment(model, loss, train_dataset, test_dataset):
    
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss=loss,
        metrics=[keras.metrics.RootMeanSquaredError()]
    )

    print("Start training the model")
    model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)
    print("Model training finished")

    output = model.evaluate(train_dataset, verbose=0, return_dict=True)
    print(output)
    # print(f"Train RMSE: {round(rmse, 3)}")

    print("Evaluating model performance...")
    output = model.evaluate(test_dataset, verbose=0, return_dict=True)
    print(output)
    # print(f"Test RMSE: {round(rmse, 3)}")



FEATURE_NAMES = [
    "age",
    "body",
    "fare",
    "parch",
    "sibsp",
]

def create_model_inputs():
    """
        For each feature create an input
    """
    inputs = {}
    for feature_name in FEATURE_NAMES:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(1,), dtype=tf.float32
        )

    return inputs


def create_baseline_model():
    inputs = create_model_inputs()
    input_values = [value for _, value in sorted(inputs.items())]

    features = keras.layers.concatenate(input_values)
    features = layers.BatchNormalization()(features)

    # Create hidden layers with deterministic weights using the Dense layer
    for units in hidden_units:
        features = layers.Dense(units, activation="sigmoid")(features)
    
    outputs = layers.Dense(units=1)(features)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


dataset_size = 4898
batch_size = 256
train_size = int(dataset_size * 0.85)
train_dataset, test_dataset = get_train_test_splits(train_size, batch_size)

num_epochs = 100
mse_loss = keras.losses.MeanSquaredError()
baseline_model = create_baseline_model()
run_experiment(baseline_model, mse_loss, train_dataset, test_dataset)


sample = 10
examples, targets = list(test_dataset.unbatch().shuffle(batch_size * 10).batch(sample))[0]
predicted = baseline_model(examples).numpy()
for idx in range(sample):
    print(f"")
