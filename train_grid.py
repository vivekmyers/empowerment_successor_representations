import numpy as np
from envs.multiagent_gridworld import MultiAgentGridWorldEnv
from envs.vectorized import vectorized
import argparse
import agents
import tqdm
import jax
import jax.numpy as jnp
import wandb
import networks.gridworld
import version

@jax.jit
def rollout_emp(key: jax.random.PRNGKey, policy: agents.Base, s: jnp.array, env):
    key, rng = jax.random.split(key)

    reward = jnp.zeros(args.num_envs)

    def step_fn(carry, _):
        s, reward, key, policy, env = carry
        obs = s

        r_ac, info = policy.next_action(obs)

        key, rng = jax.random.split(key)

        # TODO: done is now not just one boolean, but it should be [bool] with length num_humans
        s, done, human_actions = env.step_humans(s, rng) # Assume that all the humans take their steps concurrently
        env.set_state(s)

        s, r, done, _ = env.step(r_ac)
        env.set_state(s)

        reward += r

        return (s, reward, key, policy, env), (r_ac, human_actions, obs, info)

    (_, reward, _, _, _), aux = jax.lax.scan(
        step_fn, (s, reward, key, policy, env), None, length=args.max_steps
    )

    # TODO: have to rewrite this map for multiple people!
    acs_r, acs_h, trajs, infos = jax.tree_map(lambda x: jnp.moveaxis(x, 1, 0), aux) # Get the human trajectories from each step

    info = jax.tree_map(lambda x: x[0].mean(), infos)

    succ = jnp.mean(reward > 0)
    total_reward = jnp.mean(reward)

    def observe_fn(pi, aux):
        tau, ar, ah = aux
        tau = jnp.astype(tau, jnp.float32)
        return pi.observe(tau, ar, ah), None
    
    policy, _ = jax.lax.scan(observe_fn, policy, (trajs, acs_r, acs_h))

    policy, inf = policy.update()
    info.update(inf)

    return trajs, total_reward, succ, info, policy


def train(key, policy, env, itr):

    wandb.define_metric("evaluation/success", summary="mean")
    wandb.define_metric("evaluation/*", step_metric="episode")
    wandb.define_metric("buffer/*", step_metric="episode")

    for ep in tqdm.trange(itr):
        key, rng = jax.random.split(key)
        traj, ret, succ, infos, policy = rollout_emp(rng, policy, env.reset(rng), env)

        wandb.log(infos, step=policy.num_steps)
        wandb.log({"episode": ep}, step=policy.num_steps)
        wandb.log({"buffer/size": policy.buffer.buflen}, step=policy.num_steps)
        wandb.log({"buffer/index": policy.buffer.next_pos}, step=policy.num_steps)
        wandb.log({"evaluation/reward": ret}, step=policy.num_steps)
        wandb.log({"evaluation/success": succ}, step=policy.num_steps)

        if ep % args.render_freq == 0:
            traj_images = [env.image_array(s) for s in traj[0]]
            wandb.log(
                {
                    "evaluation/video": wandb.Video(np.stack(traj_images)),
                },
                step=policy.num_steps,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Gridworld for Training/Testing Empowerment")
    parser.add_argument(
        "--goal_num",
        type=str,
        default="all",
        help="Goal type: 'all' for all coordinates can be goals, 'limited' for goal set that does not contain actual goal",
    )
    parser.add_argument("--human_strategies", type=list(str), default=["NOISY_GREEDY", "RANDOM"], help="List of human strategies")
    parser.add_argument("--include_goal", action="store_true", help="Include human goal in goal set")
    parser.add_argument("--block_goal", action="store_true", help="Blocks can be on goal")
    parser.add_argument("--grid_size", type=int, default=5, help="Size of grid")
    parser.add_argument(
        "--test_case",
        type=str,
        default="corner",
        help="Test Case -- 'center' for human in center, 'corner' for human in corner, 'corner_hard' for untrapped human in corner, default is random",
    )
    parser.add_argument("--num_humans", type=int, default=2, help="Number of humans in scene")
    parser.add_argument("--num_boxes", type=int, default=4, help="Number of boxes in scene")
    parser.add_argument("--num_goals", type=int, default=2, help="Number of goals in scene")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random number generator")
    parser.add_argument("--random", action="store_true", help="Use random baseline")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of epochs")
    parser.add_argument("--num_envs", type=int, default=100, help="Number of environments to run in parallel")
    parser.add_argument("--name", default=None, help="Name of the run")

    parser.add_argument("--policy_lr", default=1e-3, type=float, help="Policy learning rate")
    parser.add_argument("--repr_lr", default=1e-4, type=float, help="Representation learning rate")
    parser.add_argument("--dual_lr", type=float, default=0.1, help="Dual learning rate")

    parser.add_argument("--repr_dim", default=32, type=int, help="Representation dimension")
    parser.add_argument("--hidden_dim", default=100, type=int, help="Hidden dimension")
    parser.add_argument("--buffer_size", default=15_000, type=int, help="Buffer size")
    parser.add_argument("--repr_buffer_size", default=15_000, type=int, help="Contrastive buffer size")

    parser.add_argument("--tau", default=0.005, type=float, help="Target network update rate for policy")
    parser.add_argument("--gamma", default=0.9, type=float, help="Discount factor for buffer")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size")

    parser.add_argument("--reward", type=str, default="dot", choices=["dot", "norm", "diff"], help="Reward function")
    parser.add_argument("--precision", default=1.0, type=float, help="Initial boltzmann constant for policy")
    parser.add_argument("--noise", default=0.2, type=float, help="Noise in human action selection")

    parser.add_argument("--update_repr_freq", default=100, type=int, help="Update frequency for representation")
    parser.add_argument("--update_policy_freq", default=100, type=int, help="Update frequency for policy")
    parser.add_argument("--update_dual_freq", type=int, default=100, help="Update frequency for dual")

    parser.add_argument("--target_entropy", type=float, default=0.6, help="Target entropy for dual")
    parser.add_argument("--cache_reward", action="store_true", help="Don't recompute reward")

    parser.add_argument("--phi_norm", action="store_true", help="Normalize phi")
    parser.add_argument("--psi_norm", action="store_true", help="Normalize psi")
    parser.add_argument("--psi_reg", default=0.0, type=float, help="Regularization on psi")
    parser.add_argument("--sample_from_target", action="store_true", help="Sample from target policy")
    parser.add_argument("--ave", action="store_true", help="Use AVE")
    parser.add_argument("--emp_rollout_len", type=int, default=5, help="Length of empowerment rollout for AVE baseline")
    parser.add_argument(
        "--emp_num_rollouts", type=int, default=20, help="Number of empowerment rollouts for AVE baseline"
    )
    parser.add_argument("--smart_features", action="store_true", help="Use smart features for AVE baseline")
    parser.add_argument('--max_steps', type=int, default=50, help='Maximum number of steps in an episode')
    parser.add_argument('--render_freq', type=int, default=20, help='Frequency of rendering')
    args = parser.parse_args()
    key = jax.random.PRNGKey(args.seed)

    wandbid = wandb.util.generate_id(4)

    if args.name is not None:
        wandbid = args.name + "-" + wandbid

    assert len(args.human_strategies) == len(args.num_humans)

    key, rng = jax.random.split(key)
    Env = vectorized(MultiAgentGridWorldEnv, args.num_envs)
    env = Env(
        human_pos=jnp.zeros(2 * args.num_humans),
        boxes_pos=jnp.zeros(2 * args.num_boxes),
        human_goals=jnp.zeros(2 * args.num_goals),
        human_strategies=args.human_strategies,
        test_case=args.test_case,
        num_boxes=args.num_boxes,
        num_humans=args.num_humans,
        num_goals=args.num_goals,
        block_goal=args.block_goal,
        grid_size=args.grid_size,
        p=1 - args.noise,
    )
    key, rng = jax.random.split(key)
    
    config = vars(args)

    config["env_name"] = "gridworld"
    config["version"] = version.__version__

    if args.random:
        policy = agents.RandomEmpowermentPolicy(rng, env.state_dim, env.nA)
        config["method"] = "random"
        wandb.init(project="multi_empowerment", config=config, id="random-" + wandbid)
    elif args.ave:
        policy = agents.AVEPolicy(
            rng,
            networks=networks.gridworld,
            s_dim=env.state_dim,
            a_dim=env.nA,
            step_human=env.step_human_action,
            policy_lr=args.policy_lr,
            hidden_dim=args.hidden_dim,
            buffer_size=args.buffer_size,
            psi_reg=args.psi_reg,
            tau=args.tau,
            gamma=args.gamma,
            update_policy_freq=args.update_policy_freq,
            batch_size=args.batch_size,
            reward=args.reward,
            precision=args.precision,
            psi_norm=args.psi_norm,
            dual_lr=args.dual_lr,
            update_dual_freq=args.update_dual_freq,
            target_entropy=args.target_entropy,
            recompute=not args.cache_reward,
            phi_norm=args.phi_norm,
            sample_from_target=args.sample_from_target,
            emp_rollout_len=args.emp_rollout_len,
            emp_num_rollouts=args.emp_num_rollouts,
            smart_features=args.smart_features,
        )
        config["method"] = "ave"
        wandb.init(project="multi_empowerment", config=config, id="ave-" + wandbid)
    else:
        policy = agents.ContrastiveEmpowermentPolicy(
            rng,
            networks=networks.gridworld,
            s_dim=env.state_dim,
            a_dim=env.nA,
            policy_lr=args.policy_lr,
            repr_lr=args.repr_lr,
            repr_dim=args.repr_dim,
            hidden_dim=args.hidden_dim,
            buffer_size=args.buffer_size,
            psi_reg=args.psi_reg,
            tau=args.tau,
            gamma=args.gamma,
            update_repr_freq=args.update_repr_freq,
            update_policy_freq=args.update_policy_freq,
            batch_size=args.batch_size,
            reward=args.reward,
            precision=args.precision,
            psi_norm=args.psi_norm,
            dual_lr=args.dual_lr,
            update_dual_freq=args.update_dual_freq,
            target_entropy=args.target_entropy,
            recompute=not args.cache_reward,
            phi_norm=args.phi_norm,
            sample_from_target=args.sample_from_target,
            repr_buffer_size=args.repr_buffer_size,
        )
        config["method"] = "contrastive"
        wandb.init(project="multi_empowerment", config=config, id="contrastive-" + wandbid)

    train(key, policy, env, args.epochs)
