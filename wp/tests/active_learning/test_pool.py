
import numpy as np
from modules.active_learning import DataPool, UnlabeledPool, LabeledPool
    


class TestLabeledPseudo:
    """
        Test functionality of pool of labeled data
    """

    def missing_targets(self):
        """Missing targets on pseudo LabeledPool"""
        pool = LabeledPool(pseudo=True)