import numpy as np
from gym.envs.mujoco import AntEnv
from boots.envs import BaseModelBasedEnv
from gym import register


class ModelBasedAntEnv(AntEnv, BaseModelBasedEnv):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],  # 13
            self.sim.data.qvel.flat,  # 14
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,  # 84
            self.get_body_com('torso')[:1],  # 1
        ])

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def mb_step(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray):
        xposbefore = states[:, -1]
        xposafter = next_states[:, -1]
        forward_reward = (xposafter - xposbefore) / self.dt

        ctrl_cost = .5 * np.square(actions).sum(axis=-1)
        contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(next_states[:, 27:111], -1, 1)), axis=-1)
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        done = ~((next_states[:, 0] >= 0.2) & (next_states[:, 0] <= 1.0))

        return reward, done


register('ModelBasedAnt-v2', entry_point=ModelBasedAntEnv, max_episode_steps=1000)
