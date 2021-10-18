import pandas as pd
import numpy as np

class Frame:
    """
        Uitiliy methods for dataframe operations across multiple dataframes.
    """

    @staticmethod
    def set_columns(frame, new_names):
        frame.columns = new_names
        return frame


    @staticmethod
    def filter(frame, exclude):

        if isinstance(exclude, list):
            column_names = frame.columns
            filtered_names = []
            for col_name in column_names:
                if col_name not in exclude:
                    filtered_names.append(col_name)

            return frame.filter(filtered_names)


        if not isinstance(exclude, dict):
            raise ValueError("Error while trying to filter the frame. Expected parameter exclude to be of type dict. Got {}".format(type(exclude)))

        filtered = frame
        for key, value in exclude.items():

            # Filter by multiple values
            if isinstance(value, list):
                for x in value:
                    selector = filtered[key] != x
                    filtered = filtered[selector]
                continue

            selector = filtered[key] != value
            filtered = filtered[selector]

        return filtered


    @staticmethod
    def split_by(composite_frame, key, force=False):
            """
                Splits a dataframe by unique values of a column.

                Parameters:
                    composite_frame (pandas.DataFrame): A pandas dataframe.
                    key (str): The key to split the dataframe by.

                Returns:
                    (list(pandas.Dataframe)) the dataframe split by column into different dataframes
            """
            frames = []

            if "float" in str(composite_frame[key].dtype) and not force:
                raise ValueError("Error in Frame.split_by(). Column to split by is of type float, aborting. Use kwarg \"force=True\" to force execution of this method.")

            values = np.unique(composite_frame[key])

            if len(values) == 1:
                return [composite_frame]
                # raise ValueError("Error in Frame.split_by(). Column {} has only one unique value {}.".format(key, values[0]))

            for unique_value in values:
                selector = composite_frame[key] == unique_value
                new_frame = composite_frame[selector]
                frames.append(new_frame)

            return frames


    @staticmethod
    def merge_by_index(frame_collection, **kwargs):
        """
            Merge multiple frame collections by array indices.

            Parameters:
                frame_collection(list(list(pandas.DataFrame))): A list of pandas dataframe lists.
                **kwargs (dict): getting passed to pandas.concat(obj, **kwargs)

            Returns:
                list(pandas.DataFrame) the frames merged by index of the inner frame lists

            Example:
                > frames = [[frame_1_1, frame_1_2], [frame_2_1, frame_2_2]]
                > merge_by_index(frames)
        """

        frames_by_index =  []
        for collection_ith in range(len(frame_collection)):

            collection = frame_collection[collection_ith]
            if not isinstance(collection, list):
                raise ValueError("Error in Frame.merge_by_index(). Expected value of type list at index {} in parameter frame collection.".format(collection_ith))

            for frame_idx in range(len(collection)):
                frame = collection[frame_idx]
                if not isinstance(frame, pd.DataFrame):
                    raise ValueError("Error in Frame.merge_by_index(). Expected frame in collection of frames at index {}.".format(frame_idx))

                if len(frames_by_index) < frame_idx+1:
                    frames_by_index.append([])

                frames_by_index[frame_idx].append(frame)

        merged_frames = []
        for idx in range(len(frames_by_index)):
            collection = frames_by_index[idx]
            merged_frames.append(pd.concat(collection, **kwargs))
        
        return merged_frames


    @staticmethod
    def mean(frames, groupby_key, ids=None):
        """
            Averages frames over all common numerical columns.

            Parameters:
                frames (pandas.DataFrames): The frames to mean data on.
                groupby_key (str): Key by which to group the frame
                ids (str|list(str)): Column names or list of column names equal over frames. Will get copied into meaned frame when passed. (default=None)

            Returns:
                (list(pandas.DataFrame)) a list of dataframes each grouped by given key and meaned.
        """

        if len(frames) < 1:
            raise ValueError("Error in Frame.mean(). Can't mean over list of <= 1 dataframe.")

        meaned = []
        for frame in frames:

            mean_frame = frame.groupby(groupby_key, as_index=False).mean()
            if ids is not None:
                mean_len = mean_frame.shape[0]
                copied_columns = Frame.get_columns(frame, ids)[:mean_len]
                Frame.update(mean_frame, copied_columns)

            meaned.append(mean_frame)
        
        return meaned


    @staticmethod
    def merge_mean_std(frame, decimals=None, mean_col="Mean", std_col="Std"):

        frame = frame.copy()
        mean_values = Frame.round_values(frame[mean_col].to_numpy(), decimals)
        std_values = Frame.round_values(frame[std_col].to_numpy(), decimals)

        zipped = zip(mean_values, std_values)
        mean_std_values = list(map(lambda x: str(x[0]) + " \u00B1 " + str(x[1]), zipped))

        mean_std_label = "Mean \u00B1 Std."

        previous_columns = frame.columns
        frame.insert(0, mean_std_label, mean_std_values)

        for column in previous_columns:
            frame = frame.drop(column, axis=1)
            
        return frame


    @staticmethod
    def update(frame, series):
        
        if isinstance(series, pd.Series):
            frame.insert(0, series.name, series)
        
        column_names = series.columns
        for idx in range(len(column_names)):
            column_name = column_names[idx]
            frame.insert(idx, column_name, series[column_name])


    @staticmethod
    def get_columns(df, names):

        if names is str:
            names = [names]
        
        return df[names]


    @staticmethod
    def transpose_index(frame, index):
        """
        
        """

        transposed = frame.copy()
        multi_index = frame.index.to_numpy()
        names = list(frame.index.names)

        index_idx = names.index(index)
        new_column = []
        new_index = []
        
        for row in multi_index:

            # Select the index corresponding to to index that should be transposed
            if isinstance(row, tuple) and len(row) > 1:
                new_column.append(row[index_idx])
                new_index.append(Frame.__from_tuple_except(row, index_idx))
            

        new_column = np.unique(new_column)
        values = frame.to_numpy()
        new_dim = int(values.shape[0]/len(new_column))
        values.reshape(tuple([new_dim] + values.shape))
    
        return transposed

    

    @staticmethod
    def __from_tuple_except(values, except_index):

        new_tuple = []
        for idx in range(len(values)):
            if idx != except_index:
                new_tuple.append(values[idx])
            
        if len(new_tuple) == 1:
            return new_tuple[0]

        return tuple(new_tuple)


    @staticmethod
    def round_values(values, decimals):

        if decimals is None:
            return values

        return np.round(values, decimals)