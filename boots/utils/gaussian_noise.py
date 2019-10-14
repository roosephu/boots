import numpy as np
import lunzi as lz


class GaussianNoise(lz.rl.BasePolicy):
    _policy: lz.rl.BasePolicy

    def __init__(self, policy: lz.rl.BasePolicy, sigma):
        self.sigma = sigma
        self._policy = policy

    def get_actions(self, states):
        actions = self._policy.get_actions(states)
        return actions + np.random.randn(*actions.shape) * self.sigma

