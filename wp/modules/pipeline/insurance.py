import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler


def mean_encode(data, column, target="charges"):
    """
        Performs mean encoding for a single column.

        Args:
            data (pd.DataFrame) The Dataframe to be used
            column (str) The column for which to perform the encoding
            target (str) The target column to be used.

        Returns:
            (tuple) (mapped data, column means)
    """

    data_means = data.groupby(column)[target].mean()
    return data[column].map(data_means), data_means




def preprocess(df, columns=[]):
    """
        Preprocess whole dataset at once.

        Args:
            df (pd.DataFrame) The dataset to be preprocessed

        Returns:
            The preprocessed dataset
    """

    new_dataset = df.copy()

    # Encode categorical variables
    # new_dataset["sex"], sex_means = mean_encode(new_dataset, "sex")
    # new_dataset["region"], region_means = mean_encode(new_dataset, "region")
    
    label_encoder = LabelEncoder()
    label_encoder.fit(new_dataset.smoker)
    new_dataset["smoker"] = label_encoder.transform(new_dataset["smoker"])

    # Normalize columns
    columns_to_scale = ["charges", "age", "bmi"]
    for column in columns_to_scale:

        data = new_dataset[[column]]
        scaler = MinMaxScaler().fit(data)
        new_dataset[column] = scaler.transform(data)



    # new_dataset["smoker"], _smoker_means = mean_encode(new_dataset, "smoker")
    new_dataset["children"], _children_means = mean_encode(new_dataset, "children")


    # Drop unnecessary columns
    drop_columns = ["sex", "region"]
    if isinstance(columns, list):
        drop_columns = drop_columns + columns
    
    else:
        logging.warning("Ignoring passed values for kwarg 'column'. Values must be of type list.")

    return new_dataset.drop(columns=drop_columns)