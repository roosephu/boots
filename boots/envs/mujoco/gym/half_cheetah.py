import numpy as np
from gym.envs.mujoco import HalfCheetahEnv
from boots.envs import BaseModelBasedEnv
from gym import register


class ModelBasedHalfCheetahEnv(HalfCheetahEnv, BaseModelBasedEnv):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def mb_step(self, states, actions, next_states):
        xposbefore = states[:, 0]
        xposafter = next_states[:, 0]
        reward_ctrl = - 0.1 * np.square(actions).sum(axis=-1)
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        return reward, np.zeros_like(reward, dtype=np.bool)

    def set_observation(self, obs):
        self.sim.data.qpos[:] = obs[:9]
        self.sim.data.qvel[:] = obs[9:]
        self.sim.forward()


register('ModelBasedHalfCheetah-v2', entry_point=ModelBasedHalfCheetahEnv, max_episode_steps=1000)
