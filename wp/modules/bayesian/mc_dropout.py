from .model import BayesModel, ModelType


class McDropout(BayesModel):
    """
        Encapsualte mc droput model.
    """

    def __init__(self, model, config):
        super(McDropout, self).__init__(model, config)


    def approx(data, **kwargs):
        pass