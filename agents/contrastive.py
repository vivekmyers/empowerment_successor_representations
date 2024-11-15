import jax
import jax.numpy as jnp
import optax
import haiku as hk
import cloudpickle as pickle
import buffers
from agents.base import Base
from simple_pytree import Pytree, static_field

# TODO: update this to only empower the first agent (index zero)

class ContrastiveEmpowermentPolicy(Pytree, Base, mutable=True):

    a_dim = static_field()
    s_dim = static_field()

    repr_fn = static_field()
    critic_fn = static_field()
    critic_opt = static_field()
    repr_opt = static_field()

    update_policy_freq = static_field()
    update_dual_freq = static_field()
    update_repr_freq = static_field()

    reward = static_field()
    psi_norm = static_field()
    phi_norm = static_field()
    psi_reg = static_field()
    gamma = static_field()
    tau = static_field()
    repr_dim = static_field()
    hidden_dim = static_field()
    recompute = static_field()
    sample_from_target = static_field()

    def __init__(
        self,
        key,
        networks,
        s_dim,
        a_dim,
        policy_lr,
        repr_lr,
        repr_dim,
        hidden_dim,
        buffer_size,
        repr_buffer_size,
        psi_reg,
        gamma,
        tau,
        update_repr_freq,
        update_policy_freq,
        batch_size,
        reward,
        precision,
        psi_norm,
        dual_lr,
        update_dual_freq,
        target_entropy,
        recompute,
        phi_norm,
        sample_from_target,
    ):

        self.key = key
        self.buffer = buffers.ContrastiveBuffer(s_dim, size=buffer_size, gamma=gamma, batch_size=batch_size)
        self.repr_buffer = buffers.ContrastiveBuffer(s_dim, size=repr_buffer_size, gamma=gamma, batch_size=batch_size)

        self.num_steps = 0
        self.a_dim = a_dim
        self.s_dim = s_dim
        s0 = jnp.zeros((1, s_dim))
        a0 = jnp.array([0])

        self.gamma = gamma
        self.psi_reg = psi_reg
        self.tau = tau
        self.update_repr_freq = update_repr_freq
        self.update_policy_freq = update_policy_freq
        self.reward = reward
        self.precision = precision
        self.psi_norm = psi_norm
        self.phi_norm = phi_norm
        self.repr_dim = repr_dim
        self.dual_lr = dual_lr
        self.update_dual_freq = update_dual_freq
        self.target_entropy = target_entropy
        self.recompute = recompute
        self.sample_from_target = sample_from_target
        self.hidden_dim = hidden_dim

        self.repr_fn = hk.without_apply_rng(hk.transform(networks.repr_fn.__get__(self, self.__class__)))
        self.critic_fn = hk.without_apply_rng(hk.transform(networks.critic_fn.__get__(self, self.__class__)))

        self.key, key = jax.random.split(self.key)
        self.repr_params = self.repr_fn.init(key, s0, a0, a0, s0)
        self.key, key = jax.random.split(self.key)
        self.critic_params = self.critic_fn.init(key, s0)
        self.target_critic_params = self.critic_params
        self.lr = repr_lr

        self.repr_opt = optax.adam(repr_lr)
        self.repr_opt_state = self.repr_opt.init(self.repr_params)

        self.critic_opt = optax.adam(policy_lr)
        self.critic_opt_state = self.critic_opt.init(self.critic_params)

    @jax.jit
    def critic_loss(self, params, target_params, s, a, r, s_next, done):
        q = self.critic_fn.apply(params, s)
        q = jnp.take_along_axis(q, a[:, None], axis=-1)[:, 0]
        q_next = self.critic_fn.apply(target_params, s_next)
        q_target = r + self.gamma * (1 - done) * jnp.max(q_next, axis=-1)
        loss = jnp.mean((q - q_target) ** 2)

        info = {
            "loss": loss,
            "q": q,
            "q_mean": q.mean(),
            "q_next": q_next,
            "q_next_mean": q_next.mean(),
            "q_target": q_target,
            "q_target_mean": q_target.mean(),
            "empowerment": r,
            "empowerment_mean": r.mean(),
            "action": a,
        }

        return loss, info

    @jax.jit
    def contrastive_loss(self, params, s, ar, ah, g):
        sa_phi, s_phi, g_psi, tau = self.repr_fn.apply(params, s, ar, ah, g)

        def infonce(phi, psi):
            logits = jnp.sum(phi[:, None, :] * psi[None, :, :], axis=-1)
            logits = logits / self.repr_dim / tau
            logits1 = jax.nn.log_softmax(logits, axis=1)
            logits2 = jax.nn.log_softmax(logits, axis=0)
            loss1 = -jnp.mean(jnp.diag(logits1))
            loss2 = -jnp.mean(jnp.diag(logits2))
            loss = loss1 + loss2
            acc1 = jnp.mean(jnp.argmax(logits1, axis=1) == jnp.arange(logits1.shape[0]))
            acc2 = jnp.mean(jnp.argmax(logits2, axis=0) == jnp.arange(logits2.shape[1]))

            l2_psi = jnp.mean(psi**2)
            l2_phi = jnp.mean(phi**2)

            loss += self.psi_reg * l2_psi**2

            info = {
                # "phi": phi,
                # "psi": psi,
                "phi_std": jnp.std(phi, axis=0).mean(),
                "psi_std": jnp.std(psi, axis=0).mean(),
                "l2_phi": l2_phi,
                "l2_psi": l2_psi,
                # "logits": logits,
                "diag": jnp.diag(logits),
                # "logits1": logits1,
                # "logits2": logits2,
                "loss1": loss1,
                "loss2": loss2,
                "acc1": acc1,
                "acc2": acc2,
                "loss": loss,
                "temp": tau,
            }

            return loss, info

        info = {}
        loss = 0.0

        g_tiled = jnp.concatenate([g_psi, g_psi], axis=0)
        s_tiled = jnp.concatenate([s_phi, sa_phi], axis=0)

        # loss_, info_ = infonce(sa_phi, g_psi)
        # loss += loss_
        # info.update({f"sa_{k}": v for k, v in info_.items()})

        # loss_, info_ = infonce(s_phi, g_psi)
        # loss += loss_
        # info.update({f"s_{k}": v for k, v in info_.items()})

        loss_, info_ = infonce(s_tiled, g_tiled)
        loss += loss_
        info.update({f"tiled_{k}": v for k, v in info_.items()})

        batch = s.shape[0]
        info["mutual_info"] = jnp.mean(info["tiled_diag"][batch:] - info["tiled_diag"][:batch])
        info["action_human"] = ah
        info["action_robot"] = ar

        return loss, info

    def compute_reward(self, sa, s, g):
        if self.reward == "dot":
            return jnp.sum((sa - s) * g, axis=-1)
        elif self.reward == "norm":
            return jnp.linalg.norm(sa, axis=-1) - jnp.linalg.norm(s, axis=-1)
        elif self.reward == "diff":
            return jnp.linalg.norm(sa - s, axis=-1)
        else:
            raise ValueError(f"Invalid reward type {self.reward}")

    @jax.jit
    def observe(self, s, a_r, a_h):
        self.num_steps += 1
        g = jnp.tile(s[-1][None], (s.shape[0], 1))
        sa_phi, s_phi, g_psi, temp = self.repr_fn.apply(self.repr_params, s, a_r, a_h, g)
        reward = self.compute_reward(sa_phi, s_phi, g_psi)
        s = jnp.astype(s, jnp.float32)
        self.buffer = self.buffer.extend(s, a_r, a_h, reward)
        self.repr_buffer = self.repr_buffer.extend(s, a_r, a_h, reward)
        return self

    def next_action(self, s):
        actor_params = self.target_critic_params if self.sample_from_target else self.critic_params
        q = self.critic_fn.apply(actor_params, s)
        adv = q - jax.scipy.special.logsumexp(q, axis=-1, keepdims=True)
        logits = self.precision * adv
        probs = jax.nn.softmax(logits, axis=-1)
        self.key, key = jax.random.split(self.key)
        a = jax.random.categorical(key, logits=logits)
        info = {
            "policy/action": a,
            "policy/action_probs": probs,
            "policy/action_logits": logits,
            "policy/action_advantages": adv,
            "policy/action_qval": q,
        }
        return a, info

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    @jax.jit
    def dual_update(self):
        self.key, key = jax.random.split(self.key)
        batch = self.buffer.sample(key)
        state = batch.state
        actor_params = self.target_critic_params if self.sample_from_target else self.critic_params
        delta, info = jax.grad(self.dual_loss_fn, has_aux=True)(self.precision, actor_params, state)
        self.precision -= self.dual_lr * delta
        self.precision = jnp.clip(self.precision, 1e-3, 100)
        info = {"dual/" + k: v for k, v in info.items()}
        info['dual/prec_grad'] = delta
        return self, info

    @jax.jit
    def update_repr(self):
        self.num_steps += 1
        self.key, key = jax.random.split(self.key)
        batch = self.repr_buffer.sample(key)

        def loss_fn(params):
            return self.contrastive_loss(
                params, batch.state, batch.action_robot, batch.action_human, batch.future_state
            )

        grad_fn = jax.grad(loss_fn, has_aux=True)
        grad, info = grad_fn(self.repr_params)
        updates, self.repr_opt_state = self.repr_opt.update(grad, self.repr_opt_state)
        self.repr_params = optax.apply_updates(self.repr_params, updates)
        info = {"repr/" + k: v for k, v in info.items()}
        return self, info

    @jax.jit
    def update_critic(self):
        self.key, key = jax.random.split(self.key)
        self.num_steps += 1
        batch = self.buffer.sample(key)
        phi_sa, phi_s, psi_g, _ = self.repr_fn.apply(
            self.repr_params, batch.state, batch.action_robot, batch.action_human, batch.future_state
        )

        rec_reward = self.compute_reward(phi_sa, phi_s, psi_g)
        if self.recompute:
            reward = rec_reward
        else:
            reward = batch.reward

        def loss_fn(params):
            return self.critic_loss(
                params,
                self.target_critic_params,
                batch.state,
                batch.action_robot,
                reward,
                batch.next_state,
                batch.done,
            )

        grad_fn = jax.grad(loss_fn, has_aux=True)
        grad, info = grad_fn(self.critic_params)
        updates, self.critic_opt_state = self.critic_opt.update(grad, self.critic_opt_state)
        self.critic_params = optax.apply_updates(self.critic_params, updates)
        self.target_critic_params = jax.tree_util.tree_map(
            lambda x, y: (1 - self.tau) * x + self.tau * y,
            self.target_critic_params,
            self.critic_params,
        )

        info.update({"rec_reward": rec_reward, "batch_reward": reward})
        info = {"critic/" + k: v for k, v in info.items()}
        return self, info
