import lunzi.nn as nn
import numpy as np


class EnsembleModel(nn.Module):
    def __init__(self, models, dim_state):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_models = len(models)
        self.dim_state = dim_state
        assert self.n_models == 1

    def forward(self, states, actions):
        return self.models[0](states, actions)

    def get_next_states(self, states, actions):
        n = len(states)
        assert n % self.n_models == 0

        perm = np.random.permutation(n).reshape(self.n_models, -1)
        next_states = np.zeros((n, self.dim_state), dtype=np.float32)
        for i in range(self.n_models):
            next_states[perm[i]] = self.models[i].get_next_states(states[perm[i]], actions[perm[i]])
        return next_states

