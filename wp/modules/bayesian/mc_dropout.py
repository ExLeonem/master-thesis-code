from .model import BayesModel, ModelType, Mode


class McDropout(BayesModel):
    """
        Encapsualte mc droput model.
    """

    def __init__(self, model, config):
        model_type = ModelType.MC_DROPUT
        super(McDropout, self).__init__(model, config, model_type=model_type)
        


    
    