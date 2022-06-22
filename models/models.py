from torchvision import models
from torch       import nn


def load_model(config):
    """
    The function of loading a model by name from a configuration file
    :param config:
    :return:
    """
    arch = config.model.arch
    num_classes = config.dataset.num_of_classes
    
    model = models.__dict__[arch](pretrained=True)
    
    if 'classifier' in model._modules.keys():
        model._modules['classifier'] 
        if isinstance( model._modules['classifier'], nn.Linear):
            in_features  = model._modules['classifier'].in_features
            out_features = num_classes
            model._modules['classifier'] = nn.Linear(in_features, out_features)
            
            nn.init.uniform_( model._modules['classifier'].weight , a=0.0, b=1.0)
            nn.init.uniform_( model._modules['classifier'].bias   , a=0.0, b=1.0)
        if isinstance( model._modules['classifier'], nn.Sequential):
            in_features  = model._modules['classifier'][-1].in_features
            out_features = num_classes
            model._modules['classifier'][-1] = nn.Linear(in_features, out_features)
            
            nn.init.uniform_( model._modules['classifier'][-1].weight , a=0.0, b=1.0)
            nn.init.uniform_( model._modules['classifier'][-1].bias   , a=0.0, b=1.0)
            
    elif 'fc' in model._modules.keys():
        in_features  = model._modules['fc'].in_features
        out_features = num_classes
        
        model._modules['fc'] = nn.Linear(in_features, out_features)
        
        nn.init.uniform_(model._modules['fc'].weight , a=0.0, b=1.0)
        nn.init.uniform_(model._modules['fc'].bias   , a=0.0, b=1.0)
    
    model.to('cuda')
    return model
