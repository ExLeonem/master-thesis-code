import .utils as butils


class BayesModel():
    """
        Base class for encapsulation of a bayesian deep learning model. 
    """

    def __init__(self, model, config):
        self.model = model
        self.config = config

    
    def approx(self, **kwargs):
        """
            Approximate predictive distribution.
        """
        pass

    

class McDropout(BayesModel):
    """
        Encapsualte mc droput model.
    """

    def __init__(self, model, config):
        super(McDropout, self).__init__(model, config)


    def approx(data, **kwargs):
        pass