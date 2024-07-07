import jax
import jax.numpy as jnp
import haiku as hk

@jax.jit
def state_features_2d(s):
    s = s.astype(jnp.int32)
    row, col = s[:2]
    box_rows = s[2::2]
    box_cols = s[3::2]
    # goal_row, goal_col = s[-2:]
    grid = jnp.zeros((5, 5, 2))
    grid = grid.at[row, col, 0].set(1)
    # grid = grid.at[goal_row, goal_col, 1].set(1)
    rows1 = jax.nn.one_hot(box_rows, 5)
    cols1 = jax.nn.one_hot(box_cols, 5)
    outer = jnp.einsum("ij,ik->jk", rows1, cols1)
    grid = grid.at[..., 1].set(outer)
    return grid

@jax.jit
def robot_action_features_2d(a_robot):
    a_robot = a_robot.astype(jnp.int32)
    ac = a_robot % 5
    boxnum = a_robot // 5
    row, col = boxnum // 5, boxnum % 5
    grid = jnp.zeros((5, 5, 5))
    grid = grid.at[row, col, ac].set(1)
    return grid


@jax.jit
def human_action_features_2d(a_human):
    a_human = a_human.astype(jnp.int32)
    grid = jnp.zeros((5, 5, 5))
    grid = grid.at[:, :, a_human].set(1)
    return grid

def repr_fn(self, s, ar, ah, g):
    f_h = jax.vmap(human_action_features_2d)
    f_r = jax.vmap(robot_action_features_2d)
    f_s = jax.vmap(state_features_2d)
    ar = f_r(ar)
    ah = f_h(ah)
    s = f_s(s)
    g = f_s(g)

    phi = hk.Sequential(
        [
            hk.Conv2D(16, 3, 1),
            jax.nn.silu,
            hk.Conv2D(32, 3, 1),
            jax.nn.silu,
            hk.Flatten(),
            hk.Linear(self.hidden_dim),
            jax.nn.silu,
            hk.Linear(self.repr_dim),
        ]
    )

    psi = hk.Sequential(
        [
            hk.Conv2D(16, 3, 1),
            jax.nn.silu,
            hk.Conv2D(32, 3, 1),
            jax.nn.silu,
            hk.Flatten(),
            hk.Linear(self.hidden_dim),
            jax.nn.silu,
            hk.Linear(self.repr_dim),
        ]
    )

    sa = jnp.concatenate([s, ar, ah], axis=-1)
    s = jnp.concatenate([s, ar, jnp.zeros_like(ah)], axis=-1)

    sa, s, g = phi(sa), phi(s), psi(g)
    if self.phi_norm:
        sa = sa / jnp.linalg.norm(sa, axis=-1, keepdims=True)
        s = s / jnp.linalg.norm(s, axis=-1, keepdims=True)
    if self.psi_norm:
        g = g / jnp.linalg.norm(g, axis=-1, keepdims=True)

    if self.phi_norm and self.psi_norm:
        temp = hk.get_parameter("temp", shape=(), init=hk.initializers.Constant(1.0))
    else:
        temp = 1.0

    return sa, s, g, temp


def critic_fn(self, s):
    if len(s.shape) == 1:
        s = state_features_2d(s)
    else:
        s = jax.vmap(state_features_2d)(s)

    q = hk.Sequential(
        [
            hk.Conv2D(16, 3, 1),
            jax.nn.silu,
            hk.Conv2D(32, 3, 1),
            jax.nn.silu,
            hk.Flatten(),
            hk.Linear(self.hidden_dim),
            jax.nn.silu,
            hk.Linear(self.a_dim),
        ]
    )

    return q(s)


if __name__ == "__main__":
    s = jnp.array([0, 1, 2, 4, 3, 3, 4, 2, 1, 1])
    a = jnp.array([0, 15, 25])
    print(jnp.moveaxis(state_features_2d(s), -1, 0))
    print(jnp.moveaxis(jax.vmap(human_action_features_2d)(a), -1, 1))

