import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.environments.overcooked.overcooked import State, Overcooked, Actions
from jaxmarl.environments.overcooked.common import DIR_TO_VEC
from flax.training import orbax_utils
import orbax
import matplotlib.pyplot as plt
import os
import argparse
import wandb
import pickle
from typing import List, Dict, Tuple
from flax.core.frozen_dict import FrozenDict
from jaxmarl.environments.overcooked.common import make_overcooked_map, OBJECT_TO_INDEX, DIR_TO_VEC
from tqdm import tqdm
import multiprocessing as mp
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from human import BCPolicy

OVERCOOKED_NAME_TO_BC_NAME = {
    "cramped_room" : "cramped_room",
    "asymm_advantages" : "asymmetric_advantages",
    "coord_ring" : "coordination_ring",
    "forced_coord" : "forced_coordination",
}


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def get_rollout(train_state, config, env):
    network = BCPolicy(env.action_space().n, activation=config["activation"])
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    init_x = jnp.zeros(env.observation_space().shape)
    init_x = init_x.flatten()

    network.init(key_a, init_x)
    network_params = train_state.params
    corrected_params = {"params": {}}
    for layer_name, layer in network_params["params"].items():
        bias = layer["bias"]
        kernel = layer["kernel"]
        if len(kernel.shape) == 3:
            kernel = kernel[0]
        if len(bias.shape) == 2:
            bias = bias[0]
        corrected_params["params"][layer_name] = {"bias": bias, "kernel": kernel}

    network_params = corrected_params

    done = False

    obs, state = env.reset(key_r)
    state_seq = [state]
    rewards = []
    while not done:
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        # obs_batch = batchify(obs, env.agents, config["NUM_ACTORS"])
        obs = {k: v.flatten() for k, v in obs.items()}

        pi_0, _ = network.apply(network_params, obs["agent_0"])
        pi_1, _ = network.apply(network_params, obs["agent_1"])

        actions = {"agent_0": pi_0.sample(seed=key_a0), "agent_1": pi_1.sample(seed=key_a1)}
        # env_act = unbatchify(action, env.agents, config["num_envs"], env.num_agents)
        # env_act = {k: v.flatten() for k, v in env_act.items()}

        # STEP ENV
        obs, state, reward, done, info = env.step(key_s, state, actions)
        done = done["__all__"]

        state_seq.append(state)
        rewards.append(reward)

    return state_seq, rewards


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def render(state_seq, env):
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

def make_train(config, obs_demos, acts_demos, obs_demos_test, acts_demos_test):
    config["NUM_UPDATES"] = config["total_timesteps"] // config["num_steps"]

    def linear_schedule(count):
        frac = 1.0 - (count // (config["update_epochs"])) / config["NUM_UPDATES"]
        return config["lr"] * frac

    def train(rng):
        # INIT NETWORK
        network = BCPolicy(6, activation=config["activation"])
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(obs_demos[0].shape)

        def _loss_fn(params, obs_batch, acts_batch):
            # Run NETWORK
            pi, _ = network.apply(params, obs_batch)
            entropy = pi.entropy().mean()

            log_prob = pi.log_prob(acts_batch)

            loss = -log_prob.mean() - config["ent_coef"] * entropy
            return loss, (loss, entropy)

        network_params = network.init(_rng, init_x)
        if config["anneal_lr"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["max_grad_norm"]), optax.adam(config["lr"], eps=1e-5))
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        # TRAIN LOOP
        def _update_step(runner_state, unused):
            train_state, rng = runner_state

            key, rng = jax.random.split(rng)
            batch_indices = jax.random.randint(rng, (config["batch_size"],), 0, acts_demos.shape[0])
            acts_batch = acts_demos[batch_indices]
            obs_batch = obs_demos[batch_indices]

            # UPDATE NETWORK
            def _update_epoch(batch_info, unused):
                train_state, obs_batch, acts_batch, rng = batch_info

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

                total_loss, grads = grad_fn(train_state.params, obs_batch, acts_batch)
                train_state = train_state.apply_gradients(grads=grads)

                update_state = (train_state, obs_batch, acts_batch, rng)
                return update_state, total_loss

            key, rng = jax.random.split(key)
            update_state = (train_state, obs_batch, acts_batch, rng)

            update_state, (train_loss, (_, train_entropy)) = jax.lax.scan(_update_epoch, update_state, None, config["update_epochs"])
            train_state = update_state[0]
            rng = update_state[-1]
            runner_state = (train_state, rng)

            test_loss, (_, test_entropy) = _loss_fn(train_state.params, obs_demos_test, acts_demos_test)
            # jax.debug.print("test loss {} train loss {}", test_loss, train_loss)

            loss_info = {"train_loss": train_loss, "train_entropy": train_entropy, "test_loss": test_loss, "test_entropy": test_entropy}
            return runner_state, loss_info

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, _rng)

        runner_state, loss_info = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "loss_info": loss_info}

    return train


def load_demos(layout_name: str, split="train") -> List:
    with open(f"human_data/human_data_state_dict_and_action_by_traj_{split}_inserted_fixed.pkl", "rb") as fb:
        all_demos = pickle.load(fb)

    return all_demos[OVERCOOKED_NAME_TO_BC_NAME[layout_name]]


def convert_old_state_to_numpy(time, wall_map, plate_pile_pos, onion_pile_pos, old_state, env, reset_state) -> Tuple[Dict[str, jnp.ndarray], State]:
    goal_pos = reset_state.goal_pos
    pot_pos = reset_state.pot_pos

    pot_status = []
    for i, pos in enumerate(pot_pos):
        found_soup = False
        for object in old_state["objects"]:
            if object["name"] == "soup" and (pos == jnp.array(object["position"])).all():
                found_soup = True

                num_onions_in_pot = len(object["_ingredients"])
                if num_onions_in_pot < 3:
                    pot_status.append(23 - num_onions_in_pot)
                elif object["is_ready"]:
                    pot_status.append(0)
                elif not object["is_cooking"]:
                    pot_status.append(20)
                else:
                    pot_status.append(20 - object["_cooking_tick"])

        if not found_soup:
            pot_status.append(23)

    onion_pos = []
    dish_pos = []
    plate_pos = []
    for object in old_state["objects"]:
        if object["name"] == "onion":
            onion_pos.append(object["position"])
        elif object["name"] == "dish":
            dish_pos.append(object["position"])
        elif object["name"] == "plate":
            plate_pos.append(object["position"])
    onion_pos = jnp.array(onion_pos)
    dish_pos = jnp.array(dish_pos)
    plate_pos = jnp.array(plate_pos)

    # Returns {"agent_0": , "agent_1": ... }
    agent_pos = jnp.array([p["position"] for p in old_state["players"]])
    agent_dir = jnp.array([p["orientation"] for p in old_state["players"]])
    agent_dir_idx = jnp.array([jnp.where(jnp.all(DIR_TO_VEC == agent_dir[0], axis=1))[0][0] for i in range(2)])

    agent_inv = jnp.array([OBJECT_TO_INDEX[p["held_object"]["name"]] if p["held_object"] is not None else OBJECT_TO_INDEX["empty"] for p in old_state["players"] if "held_object" in p])

    maze_map = make_overcooked_map(
            wall_map,
            goal_pos,
            agent_pos,
            agent_dir_idx,
            plate_pile_pos,
            onion_pile_pos,
            pot_pos,
            pot_status,
            onion_pos,
            plate_pos,
            dish_pos,
            pad_obs=True,
            num_agents=env.num_agents,
            agent_view_size=env.agent_view_size
        )

    state = State(
        agent_pos=agent_pos,
        agent_dir = agent_dir,
        agent_dir_idx = agent_dir_idx,
        agent_inv = agent_inv,
        goal_pos = goal_pos,
        pot_pos = pot_pos,
        wall_map = wall_map,
        maze_map = maze_map,
        time = time,
        terminal = False,
        glued = False
    )

    return env.get_obs(state), state


def convert_demos_to_numpy(demos: List, layout: str, key: jnp.array, split: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
    converted_states = []
    converted_actions = []

    env: Overcooked = jaxmarl.make("overcooked", layout=overcooked_layouts[layout], glue_obs=False)
    _, reset_state = env.reset(key)

    h = env.height
    w = env.width

    all_pos = jnp.arange(np.prod([h, w]), dtype=jnp.uint32)

    wall_idx = env.layout.get("wall_idx")

    occupied_mask = jnp.zeros_like(all_pos)
    occupied_mask = occupied_mask.at[wall_idx].set(1)
    wall_map = occupied_mask.reshape(h, w).astype(jnp.bool_)

    plate_pile_idx = env.layout.get("plate_pile_idx")
    plate_pile_pos = jnp.array([plate_pile_idx % w, plate_pile_idx // w], dtype=jnp.uint32).transpose()

    onion_pile_idx = env.layout.get("onion_pile_idx")
    onion_pile_pos = jnp.array([onion_pile_idx % w, onion_pile_idx // w], dtype=jnp.uint32).transpose()

    rewards = []
    for i, traj in enumerate(demos):
        state_seq = []
        rewards.append([])
        for time, x in tqdm(enumerate(traj)):
            obs, state = convert_old_state_to_numpy(time, wall_map, plate_pile_pos, onion_pile_pos, x[0], env, reset_state)
            state_seq.append(state)
            converted_states.append(obs)
        # frames = render(state_seq, env)
        #
        # wandb.log(
        #     {
        #         f"evaluation/video_{i}": wandb.Video(frames),
        #     }
        # )

        for x in traj:
            actions = []
            for act in x[1]:
                if act == "interact":
                    actions.append(Actions.interact)
                elif act[0] == 0 and act[1] == 0:
                    actions.append(Actions.stay.value)
                else:
                    actions.append(jnp.where(jnp.all(DIR_TO_VEC == jnp.array(act, dtype=jnp.int8), axis=1))[0][0].item())

            converted_actions.append(actions)

        for j in range(len(state_seq)):
            step_out = env.step_env(key, state_seq[0], {"agent_0": converted_actions[j][0], "agent_1": converted_actions[j][1]})
            rewards[-1].append(step_out[2]["agent_0"] + step_out[2]["agent_1"])
        breakpoint()
        print("Reward:", np.sum(rewards[-1]))

    print("-----\nAverage reward:", np.sum(rewards, axis=1).mean())
    breakpoint()

    converted_states = [[state["agent_0"], state["agent_1"]] for state in converted_states]
    return jnp.array(converted_states), jnp.array(converted_actions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--total_timesteps", type=int, default=5e6)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--env_name", type=str, default="overcooked")
    parser.add_argument("--anneal_lr", type=bool, default=True)
    parser.add_argument("--output", type=str, default="bc_human")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", type=str, default="cramped_room")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--recompute_obs", action="store_true")
    parser.add_argument("--test_split", type=int, default=1/4)
    parser.add_argument("--visualize_split", type=str, choices=["train", "test", "none"], default="none")
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    key = jax.random.PRNGKey(args.seed)

    folder_to_load_human_data = f"processed_human_data/{args.layout}"

    # if args.visualize_split in ["train", "test"]:
    #     wandbid = wandb.util.generate_id(4)
    #     wandb.init(project="empgrid", config=vars(args), id=f"{args.layout}_{wandbid}", group="human_true_videos")


    if args.recompute_obs or not os.path.exists(folder_to_load_human_data):
        print(f"\n\nRecomputing and writing processed data to {folder_to_load_human_data}\n\n")

        if not os.path.exists(folder_to_load_human_data):
            os.makedirs(folder_to_load_human_data)

        # wandbid = wandb.util.generate_id(4)
        # wandb.init(project="empgrid", config=vars(args), id=f"{args.layout}_{wandbid}", group="human_true_videos")

        obs_demos = {}
        acts_demos = {}
        for split in ["train", "test"]:
            key, rng = jax.random.split(key)
            demos = load_demos(args.layout, split=split)
            obs_demos_split, acts_demos_split = convert_demos_to_numpy(demos, args.layout, rng, split=split)

            obs_demos[split] = obs_demos_split
            acts_demos[split] = acts_demos_split

            jnp.save(f"{folder_to_load_human_data}/{args.layout}_acts_{split}.npy", acts_demos_split)
            jnp.save(f"{folder_to_load_human_data}/{args.layout}_obs_{split}.npy", obs_demos_split)

        print(f"Finished Saving to {folder_to_load_human_data}\n\n")
        import sys
        sys.exit(0)
    else:
        obs_demos = {}
        acts_demos = {}
        for split in ["train", "test"]:
            obs_demos[split] = jnp.load(f"{folder_to_load_human_data}/{args.layout}_obs_{split}.npy")
            acts_demos[split] = jnp.load(f"{folder_to_load_human_data}/{args.layout}_acts_{split}.npy")


    # Reshape to the right shape for each splits' obs and acts
    for split in ["train", "test"]:
        acts_demos[split] = acts_demos[split].reshape(-1)
        obs_demos[split] = obs_demos[split].reshape(acts_demos[split].shape[0], -1)


    # Re-sample the train and test
    obs_demos = jnp.vstack([obs_demos["train"], obs_demos["test"]])
    acts_demos = jnp.hstack([acts_demos["train"], acts_demos["test"]])

    test_length = int(acts_demos.shape[0]*args.test_split)
    obs_demos_test = obs_demos[:test_length]
    acts_demos_test = acts_demos[:test_length]
    obs_demos = obs_demos[test_length:]
    acts_demos = acts_demos[test_length:]

    wandbid = wandb.util.generate_id(4)
    wandb_mode = "disabled" if args.no_wandb else "online"

    wandb.init(project="empowerment", config=vars(args), id=wandbid, group="bc_" + args.layout, mode=wandb_mode)

    # set hyperparameters:
    config = {
        "env_kwargs": {
            "layout": overcooked_layouts[args.layout],
            "glue_obs": False,
            "glue_prob": 0.0,
        },
        "num_seeds": 1,
        "NUM_ACTORS": 1
    }

    config.update(vars(args))

    wandb.define_metric("test_loss", step_metric="step")
    wandb.define_metric("test_log_loss", step_metric="step")

    rngs = jax.random.split(key, config["num_seeds"])
    with jax.disable_jit(False):
        train_jit = jax.jit(jax.vmap(make_train(config, obs_demos, acts_demos, obs_demos_test, acts_demos_test)))
        out = train_jit(rngs)

        metrics = out["loss_info"]

        if not args.no_wandb:
            num_steps = metrics["test_loss"].shape[-1]
            for i in range(num_steps):
                wandb.log({"test_loss": metrics["test_loss"][0, i], "test_entropy": metrics["test_entropy"][0, i],
                           "test_log_loss": metrics["test_loss"][0, i] + config["ent_coef"]*metrics["test_entropy"][0, i],
                           "train_loss": jnp.mean(metrics["train_loss"][0], axis=1)[i],
                           "train_log_loss": jnp.mean(metrics["train_loss"][0], axis=1)[i] + config["ent_coef"]*jnp.mean(metrics["train_entropy"][0], axis=1)[i],
                           "train_entropy": jnp.mean(metrics["train_entropy"][0], axis=1)[i]})

            env = jaxmarl.make(config["env_name"], **config["env_kwargs"])

            total_rewards = []
            for _ in range(10):
                state_seq, rewards = get_rollout(out["runner_state"][0], config, env)
                total_rewards.append(sum([r["agent_0"] + r["agent_1"] for r in rewards]))

            wandb.log({"evaluation/reward": np.mean(total_rewards)})

            wandb.log(
                {
                    "evaluation/video": wandb.Video(render(state_seq, env)),
                }
            )


    if args.save:
        state = out["runner_state"][0]
        ckpt = {"model": state, "config": config}
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(ckpt)
        orbax_checkpointer.save(os.path.join(os.getcwd(), "humans", f"{args.layout}_{args.output}"), ckpt, save_args=save_args)
