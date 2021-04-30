

def switch_dropout(model, enable=True):
    """
        Switch dropout layers from eval() to train() and back.
        
        Parameters:
            - model (nn.Module) The model for which to switch the dropout layers on and off.
            - enable (boolean) Whether to turn on the training or evaulation mode.
    """
    
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            
            if enable:
                module.train()
            else:
                module.eval()