from modeling.dff_encoding.models.model_zoo import get_model
from modeling.dff_encoding.models.model_store import get_model_file
from modeling.dff_encoding.models.base import *
from modeling.dff_encoding.models.fcn import *
from modeling.dff_encoding.models.psp import *
from modeling.dff_encoding.models.encnet import *
# from modeling.dff_encoding.models.danet import *
# from modeling.dff_encoding.models.plain import *
from modeling.dff_encoding.models.casenet import *

def get_edge_model(name, **kwargs):
    from .fcn import get_fcn
    models = {
        'fcn': get_fcn,
        'psp': get_psp,
        'encnet': get_encnet,
        # 'danet': get_danet,
        # 'plain': get_plain,
        'casenet': get_casenet,
    }
    return models[name.lower()](**kwargs)
