from . import BayesModel, ModelType, Mode

class MomentPropagation(BayesModel):

    def __init__(self, model, config):
        model_type = ModelType.MOMENT_PROPAGATION
        super(MomentPropagation, self).__init__(model, config, model_type=model_type)

    
    def mean(self, predictions):
        pass


    def std(self, predictions):
        pass