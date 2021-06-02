import os, sys, csv

class Metrics:
    """
        Prepares and writes metrics into a csv file.

        Parameters:
            base_path (str): The base path where to save the metrics.
            keys (list(str)): A list of keys.
    """

    def __init__(self, base_path, keys=["accuracy", "loss"]):
        self.metric_keys = keys
        self.BASE_PATH = base_path
        self.EXT = "csv"

        # CSV Parameters
        self.delimiter = " "
        self.quotechar = "\""
        self.quoting = csv.QUOTE_MINIMAL


    def collect(self, values, keys=None):
        """
            Collect metric values from a dictionary of values.

            Parameter:
                values (dict): A collection of values collected during training

            Returns:
                (dict) A subset of metrics extracted from the values. 
        """

        # Set default keys to use
        if keys is None:
            keys = self.metric_keys

        return {key: value for key, value in values.items() if key in self.metric_keys}


    def write(self, filename, values):
        """
            Write given values into a csv file.

            Parameters:
                filename (str): The name of the file.
                values (list(dict)): A dictionary of metrics/values to write into a .csv file.
        """

        file_path = os.path.join(self.BASE_PATH, filename+"."+self.EXT)
        with open(file_path, "w", newline="") as csv_file:
            
            # Setup csv file
            file_writer = csv.DictWriter(
                csv_file, delimiter=self.delimiter, 
                quotechar=self.quotechar, quoting=self.quoting, fieldnames=self.metric_keys)

            # Create content of csv file
            file_writer.writeheader()
            for line in values:
                file_writer.writerow(line)

    
    def read(self, filename):
        """
            Read a .csv file of metrics.

            Parameters:
                filename (str): The filename to read in.

            Returns:
                (list(dict)) a list of metric values, per trained iteration.
        """

        values = []

        file_path = os.path.join(self.BASE_PATH, filename+"."+self.EXT)
        with open(file_path, "r") as csv_file:

            reader = csv.DictReader(csv_file, delimiter=self.delimiter, quotechar=self.quotechar)
            for row in reader:
                values.append(row)

        return values