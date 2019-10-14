import numpy as np
from gym.envs.mujoco.humanoid import HumanoidEnv, mass_center
from boots.envs import BaseModelBasedEnv
from gym import register


class ModelBasedHumanoidEnv(HumanoidEnv, BaseModelBasedEnv):
    """
        Observation:
            22 qpos[2:]
            23 qvel
            140 cinert
            84 cvel
            23 qfrc_actuator
            84 cfrc_ext
            1: center of mass
    """
    def _get_obs(self):
        data = self.sim.data
        center = mass_center(self.model, self.sim)
        obs = super()._get_obs()
        return np.concatenate([obs, [center]])

    def mb_step(self, states, actions, next_states):
        pos_before = states[:, -1]
        pos_after = next_states[:, -1]
        alive_bonus = 5.0
        lin_vel_cost = 1.25 * (pos_after - pos_before) / self.dt
        quad_ctrl_cost = 0.1 * np.square(actions).sum(axis=-1)
        quad_impact_cost = .5e-6 * np.square(states[:, 292:376]).sum(axis=-1)  # cfrc_ext
        quad_impact_cost = np.minimum(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        height = next_states[..., 0]
        done = (height < 1.0) | (height > 2.0)
        return reward, done


register('ModelBasedHumanoid-v2', entry_point=ModelBasedHumanoidEnv, max_episode_steps=1000)
