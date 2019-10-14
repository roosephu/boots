from typing import List
import tensorflow as tf
import numpy as np
import lunzi as lz
from lunzi import nn, rl

from boots import *
from boots.envs.batched_env import BatchedEnv
from boots.partial_envs import make_env, FLAGS as EnvFLAGS
from boots.normalizer import Normalizers
from boots.envs.virtual_env import VirtualEnv
from boots.utils import get_tf_config


class FLAGS(lz.BaseFLAGS):
    model = DeterministicModel.FLAGS
    env = EnvFLAGS
    SAC = SoftActorCritic.FLAGS
    plan = CEMPlanner.FLAGS

    n_test_samples = 10000
    n_train_samples = 1
    n_dev_samples = 1
    max_buf_size = 1000000
    n_dev_size = 2048
    n_model_ensemble = 1
    use_normalizer = True
    saving_schedule = 'exp'

    n_expl_samples = 10000

    class batch(lz.BaseFLAGS):
        n_model_grad_iters = 1
        n_policy_grad_iters = 1

        n_real = 64
        n_starts = 64
        horizon = 3

    n_iters = 1000000
    n_eval_iters = 10000
    n_save_iters = 10000

    policy_hidden_sizes = [32, 32]
    qfn_hidden_sizes = [256, 256]

    gamma = 0.99

    normalizer_decay = 1.0

    @classmethod
    def finalizer(cls):
        cls.model.n_dev_iters = 1000
        cls.SAC.batch_size = cls.batch.n_real + cls.batch.n_starts * cls.batch.horizon


@FLAGS.inject
def make_real_runner(*, n_envs, env):
    return rl.Runner(BatchedEnv([make_env] * n_envs), max_steps=env['max_steps'])


@FLAGS.inject
def evaluate(step, settings, tag, *, n_test_samples, _log, _writer):
    for runner, policy, name in settings:
        runner.reset()
        _, ep_infos = runner.run(policy, n_test_samples)
        returns = np.array([ep_info['return'] for ep_info in ep_infos])
        mean, std = np.mean(returns), np.std(returns)
        _log.info(f'# {step}, Tag = {tag}, Reward on {name} ({len(returns)} episodes): {mean:.6f} ± {std:.6f}')
        _writer.add_scalar(f'{tag}/{name}/reward/mean', mean, global_step=step)
        _writer.add_scalar(f'{tag}/{name}/reward/std', std, global_step=step)
        _writer.add_scalar(f'{tag}/{name}/reward/n', len(returns), global_step=step)


@lz.main(FLAGS)
@FLAGS.inject
def train_mb_sac(*, SAC, _log, _fs):
    tf.Session(config=get_tf_config()).__enter__()

    env = make_env()
    dim_state = int(np.prod(env.observation_space.shape))
    dim_action = int(np.prod(env.action_space.shape))
    env.verify()

    normalizers = Normalizers(dim_action=dim_action, dim_state=dim_state, decay=FLAGS.normalizer_decay)
    net_normalizer = normalizers.state if FLAGS.use_normalizer else None

    dtype = rl.gen_dtype(env, 'state action next_state reward done timeout')
    datasets = {
        'train': lz.ExtendableDataset(dtype, FLAGS.max_buf_size),
        'dev': lz.ExtendableDataset(dtype, FLAGS.n_dev_size * 2),
    }

    policy = TanhGaussianMLPPolicy(dim_state, dim_action, FLAGS.policy_hidden_sizes, normalizer=net_normalizer)
    det_policy = DetTanhPolicy(policy)
    models = []
    model_trainers = []
    for i in range(FLAGS.n_model_ensemble):
        model = DeterministicModel(dim_state, dim_action, normalizers)
        model_trainer = DeterministicModelTrainer(model, dim_state, dim_action, normalizers, datasets=datasets)
        models.append(model)
        model_trainers.append(model_trainer)
    ensemble = EnsembleModel(models, dim_state)

    virt_env = VirtualEnv(ensemble, make_env(), FLAGS.batch.n_starts)
    virt_runner = rl.Runner(virt_env, FLAGS.env.max_steps)
    # if FLAGS.batch.horizon > 0:
    #     virt_runner = VirtualRunner(dim_state, ensemble, det_policy, FLAGS.batch.horizon, dtype).build()
    qfns = nn.ModuleList([
        MLPQFunction(dim_state, dim_action, FLAGS.qfn_hidden_sizes, normalizer=net_normalizer),
        MLPQFunction(dim_state, dim_action, FLAGS.qfn_hidden_sizes, normalizer=net_normalizer),
    ])
    min_qfn = MinQFunction(dim_state, dim_action, qfns).build()

    algo = SoftActorCritic(qfns, policy, dim_state, dim_action)

    tf.get_default_session().run(tf.global_variables_initializer())

    planner = CEMPlanner(dim_state, dim_action, model, policy, min_qfn).build()
    behavior_policy = policy

    runners = {
        'collect': make_real_runner(n_envs=1),
        'dev': make_real_runner(n_envs=1),
        'eval/policy': make_real_runner(n_envs=4),
        'eval/det-policy': make_real_runner(n_envs=4),
        'eval/boots': make_real_runner(n_envs=1),
        'expand': virt_runner,
    }
    settings = [
        (runners['eval/policy'], policy, 'policy'),
        (runners['eval/det-policy'], det_policy, 'det-policy'),
        (runners['eval/boots'], planner, 'boots'),
    ]

    saver = nn.ModuleDict({'policy': policy, 'model': ensemble, 'qfns': qfns})
    _log.info(saver)

    _log.warning("Collect random trajectory...")
    datasets['train'].extend(runners['collect'].run(UniformPolicy(dim_action), FLAGS.n_expl_samples)[0])
    datasets['dev'].extend(runners['dev'].run(UniformPolicy(dim_action), FLAGS.n_expl_samples)[0])
    normalizers.update(datasets['train'][:FLAGS['n_expl_samples']])

    _log.warning("Pretraining model...")
    for model_trainer in model_trainers:
        model_trainer.pretrain()

    _log.warning("Start training...")
    for T in range(FLAGS.n_expl_samples, FLAGS.n_iters):
        if T % FLAGS.n_eval_iters == 0:
            evaluate(T, settings, 'episode')

        # collect data
        recent_train_set, ep_infos = runners['collect'].run(behavior_policy, FLAGS.n_train_samples)
        datasets['train'].extend(recent_train_set)
        datasets['dev'].extend(runners['dev'].run(policy, FLAGS.n_dev_samples)[0])

        normalizers.update(recent_train_set)

        n_policy_iters = FLAGS.batch.n_policy_grad_iters
        n_model_iters = (FLAGS.batch.n_model_grad_iters + n_policy_iters - 1) // n_policy_iters
        for _ in range(n_policy_iters):
            for _ in range(n_model_iters):
                for model_trainer in model_trainers:
                    model_trainer.step()

            real_samples = datasets['train'].sample(FLAGS.batch.n_real)
            if FLAGS.batch.horizon > 0:
                fake_starts = datasets['train'].sample(FLAGS.batch.n_starts)['state']
                runners['expand'].set_state(fake_starts)
                rollout_policy = policy
                fake_samples = runners['expand'].run(rollout_policy, FLAGS.batch.n_starts * FLAGS.batch.horizon)[0]
                samples = np.concatenate([real_samples, fake_samples]).astype(fake_samples.dtype).view(lz.Dataset)
            else:
                samples = real_samples

            algo.train(samples)

        if T % FLAGS.n_save_iters == 0:
            t = T // FLAGS.n_save_iters
            _fs.save('$LOGDIR/final.npy', saver.state_dict())
            if FLAGS.saving_schedule == 'exp' and t & (t - 1) == 0:
                _fs.save(f'$LOGDIR/stage-{T}.npy', saver.state_dict())
            elif FLAGS.saving_schedule == 'every':
                _fs.save(f'$LOGDIR/stage-{T}.npy', saver.state_dict())


if __name__ == '__main__':
    # with tf.device('/device:GPU:0'):
    train_mb_sac()
