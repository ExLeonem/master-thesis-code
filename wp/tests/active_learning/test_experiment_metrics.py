import os, shutil
import pytest
from modules.active_learning import experiment_metrics


class TestUtilities:

    def test_base(self):
        pass



METRICS_PATH = os.path.join(os.getcwd(), "metrics")


class TestExperimentMetricsReadWrite:

    def setup_method(self):
        """
            Build Metrics directory to perform tests.
        """
        if not os.path.exists(METRRICS_PATH):
            os.mkdir(METRICS_PATH)

    
    def teardown_method(self):
        """
            Clear test directory from metrics.
        """
        if os.path.exists(METRICS_PATH):
            shutil.rmtree(METRICS_PATH)

        
    