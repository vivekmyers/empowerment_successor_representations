import argparse
import agents
import tqdm
import jax
import jax.numpy as jnp
import wandb
import networks.overcooked
import jaxmarl
import human
import os
import orbax
import orbax.checkpoint
from jaxmarl.environments.overcooked import Overcooked, overcooked_layouts
from jaxmarl.environments.overcooked.overcooked import Actions
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
import numpy as np
import haiku as hk
from functools import reduce
import version


def step_human(obs, rng):
    global human_params
    pi, val = human_network.apply(human_params, obs)
    a = pi.sample(seed=rng)
    return a


@jax.jit
def compute_empowerment(s, rng):
    keys = hk.PRNGSequence(rng)
    s = jax.tree_map(lambda x: jnp.repeat(x[None], args.emp_num_rollouts, axis=0), s)
    for _ in range(args.emp_rollout_len):
        rng = jax.random.split(next(keys), args.emp_num_rollouts)
        ac = jax.random.randint(next(keys), (args.emp_num_rollouts,), 0, env.action_space().n)
        actions = {"agent_0": jnp.repeat(Actions.stay, args.emp_num_rollouts), "agent_1": ac}
        obs, state, _, _, _ = jax.vmap(env.step)(rng, s, actions)
    obs = obs["agent_1"]
    obs = obs.reshape((obs.shape[0], -1))
    if args.smart_features:
        return jnp.log(jnp.var(obs, axis=0).mean(axis=-1) + 1e-3)
    else:
        svec = obs.flatten()
        return jnp.log(jnp.var(svec))

@jax.jit
def rollout_emp(key: jax.random.PRNGKey, policy: agents.Base):
    key, rng = jax.random.split(key)

    rngs = jax.random.split(rng, args.num_envs)
    obs, state = jax.vmap(env.reset)(rngs)
    # breakpoint()
    obs = jax.vmap(lambda x: {"agent_0": x["agent_0"].reshape(-1), "agent_1": x["agent_1"]})(obs)
    # obs = jax.tree_map(lambda x: x.reshape((args.num_envs, -1)), obs)
    reward = jnp.zeros(args.num_envs)

    def step_fn(carry, _):
        obs, state, reward, key, policy = carry
        state0 = state
        r_ac, info = policy.next_action(obs["agent_0"])

        key, rng = jax.random.split(key)
        h_ac = step_human(obs["agent_1"], rng)

        key, rng = jax.random.split(key)
        rngs = jax.random.split(rng, args.num_envs)
        obs, state, r, done, _ = jax.vmap(env.step)(rngs, state, {"agent_0": r_ac, "agent_1": h_ac})

        obs = jax.vmap(lambda x: {"agent_0": x["agent_0"].reshape(-1), "agent_1": x["agent_1"]})(obs) #
        # obs = jax.tree_map(lambda x: x.reshape((args.num_envs, -1)), obs)

        reward += r["agent_1"]
        if args.ave:
            key, rng = jax.random.split(key)
            rng = jax.random.split(rng, args.num_envs)
            emp = jax.vmap(compute_empowerment)(state, rng)
        else:
            emp = jnp.zeros_like(reward)

        return (obs, state, reward, key, policy), (r_ac, h_ac, obs["agent_0"], state0, emp, info)

    (_, _, reward, key, policy), aux = jax.lax.scan(
        step_fn, (obs, state, reward, key, policy), None, length=args.max_steps
    )

    acs_r, acs_h, trajs, states, emps, infos = jax.tree_map(lambda x: jnp.moveaxis(x, 1, 0), aux)

    info = jax.tree_map(lambda x: x[0].mean(), infos)

    succ = jnp.mean(reward > 0)
    total_reward = jnp.mean(reward)

    seq = jax.tree_map(lambda x: x[0], states)
    leaves, treedef = jax.tree_util.tree_flatten(seq)
    seq = [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]

    def observe_fn(policy, aux):
        tau, ar, ah, emp = aux
        if args.ave:
            tau = jnp.astype(tau, jnp.float32)
            return policy.observe(tau, ar, emp), None
        else:
            return policy.observe(tau, ar, ah), None
    
    policy, _ = jax.lax.scan(observe_fn, policy, (trajs, acs_r, acs_h, emps))


    policy, inf = policy.update()
    info.update(inf)

    return trajs, total_reward, succ, seq, info, policy


def render(state_seq):
    padding = env.agent_view_size - 2  # show

    def get_frame(state):
        grid = np.asarray(state.maze_map[padding:-padding, padding:-padding, :])
        # Render the state
        frame = OvercookedVisualizer._render_grid(
            grid, tile_size=32, highlight_mask=None, agent_dir_idx=state.agent_dir_idx, agent_inv=state.agent_inv
        )
        return frame

    frame_seq = np.stack([get_frame(state) for state in state_seq])
    frame_seq = np.moveaxis(frame_seq, -1, 1)

    return frame_seq


def train(key, policy, env, itr):

    wandb.define_metric("evaluation/success", summary="mean")
    wandb.define_metric("evaluation/*", step_metric="episode")
    wandb.define_metric("buffer/*", step_metric="episode")

    key, rng = jax.random.split(key)

    baselines = []
    for _ in tqdm.trange(20):
        _, ret, _, _, _, _ = rollout_emp(rng, random_policy)
        baselines.append(ret)
    baselines = jnp.array(baselines)
    baseline = jnp.mean(baselines)
    reward_ema = None

    for ep in tqdm.trange(itr):
        key, rng = jax.random.split(key)
        traj, ret, succ, state_seq, infos, policy = rollout_emp(rng, policy)

        wandb.log(infos, step=policy.num_steps)

        if reward_ema is None:
            reward_ema = ret
        reward_ema = 0.99 * reward_ema + 0.01 * ret

        wandb.log({"episode": ep}, step=policy.num_steps)
        wandb.log({"buffer/size": policy.buffer.buflen}, step=policy.num_steps)
        wandb.log({"buffer/index": policy.buffer.next_pos}, step=policy.num_steps)

        wandb.log({"evaluation/reward": ret}, step=policy.num_steps)
        wandb.log({"evaluation/reward_ema": reward_ema}, step=policy.num_steps)
        wandb.log({"evaluation/success": succ}, step=policy.num_steps)
        wandb.log({"evaluation/baseline": baseline}, step=policy.num_steps)
        wandb.log({"evaluation/delta": ret - baseline}, step=policy.num_steps)
        wandb.log({"evaluation/delta_ema": reward_ema - baseline}, step=policy.num_steps)

        if ep % args.render_freq == 0:
            wandb.log(
                {
                    "evaluation/video": wandb.Video(render(state_seq)),
                },
                step=policy.num_steps,
            )

        # if ep % 100 == 0:
        #     print(f"Step {ep}/{itr}")
        #     policy.save(f"checkpoints/contrastive_{ep}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gridworld for Empowerment")
    parser.add_argument(
        "--goal_num",
        type=str,
        default="all",
        help="Goal type: 'all' for all coordinates can be goals, 'limited' for goal set that does not contain actual goal",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for random number generator")
    parser.add_argument("--random", action="store_true", help="Use random baseline")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of epochs")
    parser.add_argument("--num_envs", type=int, default=100, help="Number of environments to run in parallel")
    parser.add_argument("--name", default=None, help="Name of the run")

    parser.add_argument("--policy_lr", default=1e-5, type=float, help="Policy learning rate")
    parser.add_argument("--repr_lr", default=3e-5, type=float, help="Representation learning rate")
    parser.add_argument("--dual_lr", type=float, default=0.1, help="Dual learning rate")

    parser.add_argument("--repr_dim", default=32, type=int, help="Representation dimension")
    parser.add_argument("--hidden_dim", default=100, type=int, help="Hidden dimension")
    parser.add_argument("--buffer_size", default=200_000, type=int, help="Buffer size")

    parser.add_argument("--tau", default=1e-5, type=float, help="Target network update rate for policy")
    parser.add_argument("--gamma", default=0.8, type=float, help="Discount factor for buffer")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size")

    parser.add_argument("--reward", type=str, default="dot", choices=["dot", "norm", "diff"], help="Reward function")
    parser.add_argument("--precision", default=1.0, type=float, help="Initial boltzmann constant for policy")

    parser.add_argument("--update_repr_freq", default=100, type=int, help="Update frequency for representation")
    parser.add_argument("--update_policy_freq", default=30, type=int, help="Update frequency for policy")
    parser.add_argument("--update_dual_freq", type=int, default=100, help="Update frequency for dual")

    parser.add_argument("--target_entropy", type=float, default=0.9, help="Target entropy for dual")
    parser.add_argument("--cache_reward", action="store_true", help="Don't recompute reward")

    parser.add_argument("--phi_norm", action="store_true", help="Normalize phi")
    parser.add_argument("--psi_norm", action="store_true", help="Normalize psi")
    parser.add_argument("--psi_reg", default=0.0, type=float, help="Regularization on psi")
    parser.add_argument("--sample_from_target", action="store_true", help="Sample from target policy")
    parser.add_argument("--ave", action="store_true", help="Use AVE")
    parser.add_argument("--emp_rollout_len", type=int, default=5, help="Length of empowerment rollout for AVE baseline")
    parser.add_argument("--emp_num_rollouts", type=int, default=20, help="Number of empowerment rollouts for AVE baseline")
    parser.add_argument("--smart_features", action="store_true", help="Use smart features for AVE baseline")
    parser.add_argument("--max_steps", type=int, default=400, help="Max steps in rollout")
    parser.add_argument("--render_freq", type=int, default=20, help="Frequency of rendering")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to restore")
    parser.add_argument(
        "--layout",
        type=str,
        default="cramped_room",
        help="Layout of the overcooked environment",
        choices=overcooked_layouts.keys(),
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb")
    args = parser.parse_args()

    wandb_mode = "disabled" if args.no_wandb else "online"
    key = jax.random.PRNGKey(args.seed)

    wandbid = wandb.util.generate_id(4)

    if args.name is not None:
        wandbid = args.name + "-" + wandbid

    wandbid = "overcooked-" + wandbid

    key, rng = jax.random.split(key)
    env: Overcooked = jaxmarl.make("overcooked", layout=overcooked_layouts[args.layout])
    viz = OvercookedVisualizer()

    # human_network = human.BCPolicy(env.action_space().n, activation="tanh")
    human_network = human.TorchConversionActorCritic()
    init_x = jnp.zeros((1, *env.observation_space().shape))
    initial_human_params = human_network.init(rng, init_x)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    if args.checkpoint is None:
        checkpoint = os.path.join(os.getcwd(), f"humans/{args.layout}-{args.seed}")
    else:
        checkpoint = args.checkpoint
    raw_restored = orbax_checkpointer.restore(checkpoint)

    human_actor_params = raw_restored["model"][
        "params"
    ]  # np.load(config["HUMAN_ACTOR_CHECKPOINT"], allow_pickle=True).item()
    corrected_params = {"params": {}}
    for layer_name, layer in human_actor_params["params"].items():
        bias = layer["bias"]
        kernel = layer["kernel"]

        initial_bias = initial_human_params["params"][layer_name]["bias"]
        initial_kernel = initial_human_params["params"][layer_name]["kernel"]
        if bias.shape != initial_bias.shape:
            bias = bias.reshape(initial_bias.shape)
        if kernel.shape != initial_kernel.shape:
            kernel = kernel.reshape(initial_kernel.shape)
        # if len(kernel.shape) == 3:
        #     kernel = kernel[0]
        # if len(bias.shape) == 2:
        #     bias = bias[0]
        corrected_params["params"][layer_name] = {"bias": bias, "kernel": kernel}

    human_params = corrected_params
    key, rng = jax.random.split(key)

    policy_kwargs = dict(
        update_policy_freq=args.update_policy_freq,
        policy_lr=args.policy_lr,
        hidden_dim=args.hidden_dim,
        tau=args.tau,
        psi_reg=args.psi_reg,
        gamma=args.gamma,
        batch_size=args.batch_size,
        precision=args.precision,
        update_dual_freq=args.update_dual_freq,
        target_entropy=args.target_entropy,
        recompute=not args.cache_reward,
        buffer_size=args.buffer_size,
        reward=args.reward,
        dual_lr=args.dual_lr,
        sample_from_target=args.sample_from_target,
        psi_norm=args.psi_norm,
        phi_norm=args.phi_norm,
    )
    config = vars(args)

    config["env_name"] = "overcooked"
    # s_dim = 520
    s_dim = reduce(lambda x, y: x * y, env.observation_space().shape, 1)
    config["version"] = version.__version__

    random_policy = agents.RandomEmpowermentPolicy(rng, s_dim, env.action_space().n)
    if args.random:
        policy = random_policy
        config["method"] = "random"
        wandb.init(project="empowerment", config=config, id="random-" + wandbid, mode=wandb_mode)
    elif args.ave:
        policy = agents.SoftDQN(
            rng,
            networks=networks.overcooked,
            smart_features=args.smart_features,
            a_dim=env.action_space().n,
            s_dim=s_dim,
            **policy_kwargs,
        )
        config["method"] = "ave"
        wandb.init(project="empowerment", config=config, id="ave-" + wandbid, mode=wandb_mode)
    else:
        policy = agents.ContrastiveEmpowermentPolicy(
            rng,
            networks=networks.overcooked,
            s_dim=s_dim,
            a_dim=env.action_space().n,
            repr_lr=args.repr_lr,
            repr_dim=args.repr_dim,
            update_repr_freq=args.update_repr_freq,
            **policy_kwargs,
        )
        config["method"] = "contrastive"
        wandb.init(project="empowerment", config=config, id="contrastive-" + wandbid, mode=wandb_mode)

    train(key, policy, env, args.epochs)
