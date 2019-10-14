import numpy as np
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv, VecEnv, Pipe, CloudpickleWrapper, Process
import lunzi as lz


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                if done:
                    ob = env.reset()
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'render':
                remote.send(env.render(mode='rgb_array'))
            elif cmd == 'simulate':
                states, actions = data

                next_states = []
                for state, action in zip(states, actions):
                    env.unwrapped.sim.set_state(state)
                    next_state, reward, done, info = env.step(action)
                    next_states.append(next_state)
                remote.send(np.array(next_states))

            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class _SubprocVecEnv(SubprocVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        Arguments:
        env_fns: iterable of callables -  functions that create environments to run in subprocesses.
        Need to be cloud-pickleable
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.viewer = None
        self.specs = [f().spec for f in env_fns]
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def simulate(self, states, actions):
        n = len(states)
        n_envs = len(self.remotes)
        states = states.reshape(n_envs, n // n_envs, -1)
        actions = actions.reshape(n_envs, n // n_envs, -1)

        for remote, batch_states, batch_actions in zip(self.remotes, states, actions):
            remote.send(('simulate', (batch_states, batch_actions)))

        results = [remote.recv() for remote in self.remotes]
        return np.vstack(results)


class FLAGS(lz.BaseFLAGS):
    n_envs = 10


class OracleModel:
    FLAGS = FLAGS

    @FLAGS.inject
    def __init__(self, make_env, n_envs):
        self._envs = _SubprocVecEnv([make_env] * n_envs)
        self.n_envs = n_envs

    def get_next_states(self, states, actions):
        assert len(states) == len(actions)
        return self._envs.simulate(states, actions)
