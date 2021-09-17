import pandas as pd
import numpy as np

class Frame:
    """
        Uitiliy methods for dataframe operations across multiple dataframes.
    """


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
                raise ValueError("Error in Frame.split_by(). Column {} has only one unique value {}.".format(key, values[0]))

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

        if len(frames) <= 1:
            raise ValueError("Error in Frame.mean(). Can't mean over list of <= 1 dataframe.")

        meaned = []
        for frame in frames:
            
            mean_frame = frame.groupby(groupby_key).mean()
            meaned.append(mean_frame)
        
        return meaned