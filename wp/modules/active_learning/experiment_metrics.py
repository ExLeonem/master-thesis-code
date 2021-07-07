import os, sys
import csv, json


class ExperimentSuitMetrics:
    """
        Handles metrics of experiments.

        Example content of the meta file.

            {
                'dataset': {
                    'name': 'mnist',
                    'path': '...',
                    'train_size': 0.75,
                    'test_size': 0.25,
                    'val_size': 0.15
                },
                'experiments': [
                    {
                        'model': 'mc_dropout',
                        'query_fn': 'max_entropy',
                        'filename': 'filename',
                        'params': {
                            'iterations': 100,
                            'step_size': 10,
                            'initial_size': 100,
                            'fit': {
                                'optimizer': 'adam',
                                'learning_rate': 0.15,
                                'loss': 'sparse_categorical_crossentropy'
                            }
                        }
                    }
                ]
            }
    """


    def __init__(self, base_path):
        self.BASE_PATH = base_path
        self.META_FILE = os.path.join(base_path, ".meta.json")
        self.__setup_dir(BASE_PATH)

        # Keep track of written experiment metrics
        self.experiment_file = {}
        self.last_opened = None

        # CSV Parameters
        self.delimiter = " "
        self.quotechar = "\""
        self.quoting = csv.QUOTE_MINIMAL


    def __setup_dir(self, path):
        # Try to create directory if non existen
        if not os.path.exists(path):
            os.mkdir(path)
        
        # Create non-existent meta.json file
        if not os.path.exists(self.META_FILE):
            base_content = {"experiments": []}
            self.write_meta(base_content)


    def add_dataset_info(self, name, path, train_size, test_size=None, val_size=None):
        meta = self.read_meta()
        meta["dataset"] = {
            "name": name,
            "path": path,
            "train_size": train_size
        }

        if test_size is not None:
            meta["dataset"]["test_size"] = test_size

        if val_size is not None:
            meta["dataset"]["val_size"] = val_size

        
        self.write_meta(meta)


    def add_meta(self, filename, model_name, query_fn, **params):
        """
            
        """

        meta = self.read_meta()
        experiments = meta["experiments"]

        experiments.append({
            "model": model_name,
            "query_fn", query_fn,
            "file_name": filename,
            "params": params
        })

        meta["experiments"] = experiments
        self.write_meta(meta)


    # ----------
    # Read/Write utilties
    # -------------------------

    def write_meta(self, content):
        """

        """

        with open(self.META_FILE, "w") as json_file:
            json_file.write(json.dumps(content))


    def read_meta(self):
        """

        """
        
        content = {}
        with open(self.META_FILE, "r") as json_file:
            content = json_file.read()

        return json.loads(content)

    
    def write_line(self, experiment_name, values, meta=None, keys=None, **kwargs):
        """
            Parameter:
                experiment_name (str): The name of the experiment performed.
        """

        experiments = self.experiment_file.keys()
        if experiment_name not in experiments:
            self.experiment_file[experiment_name] = 
