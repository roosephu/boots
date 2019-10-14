import numpy as np
import gym

import lunzi as lz

import boots.envs.mujoco.gym


class FLAGS(lz.BaseFLAGS):
    id = 'HalfCheetah-v2'
    max_steps = 1000


class RescaleAction(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lo = env.action_space.low
        self.hi = env.action_space.high
        self.action_space = gym.spaces.Box(low=np.full_like(self.lo, -1.), high=np.full_like(self.hi, +1.))

    def action(self, action):
        return self.lo + (action + 1.) * 0.5 * (self.hi - self.lo)

    def reverse_action(self, action):
        return (action - self.lo) / np.maximum(self.hi - self.lo, 1.e-9) * 2. - 1.

    def mb_step(self, states, actions, next_states):
        return self.env.mb_step(states, self.action(actions), next_states)


@FLAGS.inject
def make_env(*, id: str, _rng):
    env = gym.make(id).unwrapped
    env = RescaleAction(env)
    env.seed(_rng.randint(0, 10**9))
    return env
