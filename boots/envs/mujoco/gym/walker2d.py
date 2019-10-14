import numpy as np
from gym.envs.mujoco import Walker2dEnv
from boots.envs import BaseModelBasedEnv
from gym import register


class ModelBasedWalker2dEnv(Walker2dEnv, BaseModelBasedEnv):
    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos, qvel]).ravel()

    def mb_step(self, states, actions, next_states):
        posbefore = states[:, 0]
        posafter, height, ang = next_states[:, :3].T
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(actions).sum(axis=-1)
        done = ~((height > 0.8) & (height < 2.0) & (ang > -1.0) & (ang < 1.0))
        return reward, done

    def set_observation(self, obs):
        self.sim.data.qpos[:] = obs[:9]
        self.sim.data.qvel[:] = obs[9:]
        # self.model._compute_subtree()
        self.sim.forward()


register('ModelBasedWalker2d-v2', entry_point=ModelBasedWalker2dEnv, max_episode_steps=1000)
