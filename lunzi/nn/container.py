from typing import Dict, List, Any
from .module import Module
from .parameter import Parameter

_dict_methods = ['__setitem__', '__delitem__', '__len__', '__iter__', '__contains__',
                 'update', 'keys', 'values', 'items', 'clear', 'pop']
_list_methods = ['__len__', '__iter__']


class ModuleDict(Module, dict):  # use dict for auto-complete
    """
        Essentially this exposes some methods of `Module._modules`.
    """
    def __init__(self, modules: Dict[Any, Module] = None):
        super().__init__()
        for method in _dict_methods:  # can we use a meta class for that?
            setattr(self, method, getattr(self._modules, method))
        if modules:
            self.update(modules)

    def __getitem__(self, item):
        return self._modules[item]

    def forward(self):
        raise RuntimeError("ModuleDict is not callable")


# Do we need a factory for it?
class ParameterDict(Module, dict):
    def __init__(self, parameters: Dict[Any, Parameter] = None):
        super().__init__()
        for method in _dict_methods:
            setattr(self, method, getattr(self._parameters, method))
        if parameters:
            self.update(parameters)

    def forward(self):
        raise RuntimeError("ParameterDict is not callable")


class ModuleList(Module, list):
    def __init__(self, modules: List[Module] = None):
        super().__init__()
        self._sync_modules = []
        if modules:
            self.extend(modules)

    def __len__(self):
        return len(self._sync_modules)

    def __getitem__(self, item):
        return self._sync_modules[item]

    def __iter__(self):
        return iter(self._sync_modules)

    def _sync(self):
        self._modules = {i: module for i, module in enumerate(self._sync_modules)}

    def append(self, module):  # for efficiency
        self._modules[len(self)] = module
        self._sync_modules.append(module)

    def extend(self, modules):
        for module in modules:
            self.append(module)

    def __delitem__(self, idx):
        del self._sync_modules[idx]
        self._sync()

    def forward(self):
        raise RuntimeError("ModuleList in not callable")

