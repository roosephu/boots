import abc
from typing import Union
import lunzi.nn as nn


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def get_actions(self, states):
        pass

    def reset(self, indices=None):  # for stateful policies
        pass

    def step(self):  # for stateful policies
        pass


BaseNNPolicy = Union[BasePolicy, nn.Module]  # should be Intersection, see PEP544
