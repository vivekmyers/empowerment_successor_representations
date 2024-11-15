import jax
import jax.numpy as jnp
import optax
import haiku as hk
import functools
import buffers
from agents.base import Base
from simple_pytree import static_field
from simple_pytree import Pytree

# TODO: update this to empower only the first agent (index zero)

class AVEPolicy(Pytree, Base, mutable=True):

    a_dim = static_field()
    s_dim = static_field()

    critic_fn = static_field()
    critic_opt = static_field()

    update_policy_freq = static_field()
    update_dual_freq = static_field()

    emp_rollout_len = static_field()
    emp_num_rollouts = static_field()
    smart_features = static_field()
    step_human = static_field()
    hidden_dim = static_field()
    sample_from_target = static_field()
    phi_norm = static_field()
    psi_norm = static_field()

    reward = static_field()
    recompute = static_field()

    def __init__(
        self,
        key,
        networks,
        a_dim,
        s_dim,
        step_human,
        policy_lr,
        hidden_dim,
        buffer_size,
        psi_reg,
        gamma,
        tau,
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
        emp_rollout_len,
        emp_num_rollouts,
        smart_features,
    ):

        self.key = key
        self.buffer = buffers.ContrastiveBuffer(s_dim, size=buffer_size, gamma=gamma, batch_size=batch_size)
        self.hidden_dim = hidden_dim
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.num_steps = 0

        self.critic_fn = hk.without_apply_rng(hk.transform(networks.critic_fn.__get__(self, self.__class__)))

        s0 = jnp.zeros((1, s_dim))
        a0 = jnp.array([0])

        self.key, key = jax.random.split(self.key)
        self.critic_params = self.critic_fn.init(key, s0)
        self.target_critic_params = self.critic_params

        self.critic_opt = optax.adam(policy_lr)
        self.critic_opt_state = self.critic_opt.init(self.critic_params)

        self.gamma = gamma
        self.psi_reg = psi_reg
        self.tau = tau
        self.update_policy_freq = update_policy_freq
        self.reward = reward
        self.precision = precision
        self.psi_norm = psi_norm
        self.dual_lr = dual_lr
        self.update_dual_freq = update_dual_freq
        self.target_entropy = target_entropy
        self.recompute = recompute
        self.phi_norm = phi_norm
        self.sample_from_target = sample_from_target
        self.emp_rollout_len = emp_rollout_len
        self.emp_num_rollouts = emp_num_rollouts
        self.smart_features = smart_features
        self.step_human = step_human

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
    def compute_reward(self, s):
        s = jnp.repeat(s[..., None, :], self.emp_num_rollouts, axis=-2)
        for _ in range(self.emp_rollout_len):
            self.key, key = jax.random.split(self.key)
            a = jax.random.randint(key, s.shape[:-1], 0, self.a_dim)
            s = jnp.vectorize(self.step_human, signature="(n),()->(n)")(s, a)
        if self.smart_features:
            return jnp.log(jnp.var(s, axis=-2).mean(axis=-1) + 1e-3)
        else:
            svec = s.reshape(list(s.shape[:-2]) + [-1])
            return jnp.log(jnp.var(svec, axis=-1))

    @jax.jit
    def observe(self, s, a_r, a_h):
        self.num_steps += 1
        reward = self.compute_reward(s)
        self.buffer = self.buffer.extend(s, a_r, a_h, reward)
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
        return self, info

    @jax.jit
    def update_critic(self):
        self.key, key = jax.random.split(self.key)
        self.num_steps += 1
        batch = self.buffer.sample(key)

        rec_reward = self.compute_reward(batch.state)
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
