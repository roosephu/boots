__version__ = '0.0.1'

from typing import Union

from .dummy import dummy
from .file_storage import FileStorage
from .experiment import init, close, main, get_logger, SummaryWriter, Logger
from .injector import inject


log: Logger = get_logger('lunzi')
fs: FileStorage = FileStorage()
writer: Union[dummy, SummaryWriter] = dummy
features = []
info = {
    'lunzi': {
        'features': features,
        '__version__': __version__,
    }
}

try:
    from .stubs import Tensor
    import lunzi.nn
    from .multi_layer_perception import MultiLayerPerceptron
    features.append('tf-nn')
except ImportError:
    pass

try:
    from . import rl
    features.append('rl')
except ImportError as e:
    pass

from .serialization import save, load
from .dataset import BaseDataset, Dataset, ExtendableDataset
from .sampler import BaseSampler, BatchSampler
from . import debug
from .typing import *
from .profiler import *

log.critical(f'lunzi features: [{", ".join(features)}]')
