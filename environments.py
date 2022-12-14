import jax as jx
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial, reduce
import operator
from math import ceil
import gymnax
import gym
from copy import deepcopy

# Simple example gridworld environment in JAX
class OpenGrid:
    def __init__(self, grid_size=10, spontaneous_termination=True, teleport_on_termination=True):
        #0: no-op, 1: up, 2: left, 3: down, 4: right
        self.move_map = jnp.asarray([[0, 0], [-1,0], [0,-1], [1,0], [0,1]])

        self._num_actions = 5
        self.grid_size = grid_size
        self.channels ={
            'player':0
        }

        self.goal = jnp.asarray([grid_size-1,grid_size-1])

        #1/10th as often as the optimal time to solve the worst case layout for gridsize
        if(spontaneous_termination):
            self.spontaneous_goal_probability=0.1/self.grid_size
        else:
            self.spontaneous_goal_probability=0.0

        self.teleport_on_termination = teleport_on_termination

    @partial(jit, static_argnums=(0,))
    def step(self, key, env_state, action):
        # print(env_state)
        pos = env_state
        terminal = False

        # Reset the step after if goal is reached, so agent sees the state where pos==goal
        terminal = jnp.array_equal(pos, self.goal)

        # punish agent for each step until termination
        reward = -1

        # Move if the new position is on the grid
        pos = jnp.clip(pos+self.move_map[action], 0, self.grid_size-1)

        # With small probability, teleport to goal
        key, subkey = jx.random.split(key)
        spontanteous_goal = jx.random.bernoulli(subkey, p=self.spontaneous_goal_probability)
        if(self.teleport_on_termination):
            pos = jnp.where(spontanteous_goal, self.goal, pos)
        else:
            terminal = jnp.logical_or(terminal, spontanteous_goal)

        env_state = pos

        return env_state, self.get_observation(env_state), reward, terminal, {}

    @partial(jit, static_argnums=(0,))
    def reset(self, key):
        key, subkey = jx.random.split(key)
        pos = jx.random.choice(subkey, self.grid_size, (2,))
        env_state = pos
        return env_state, self.get_observation(env_state)

    @partial(jit, static_argnums=(0,))
    def get_observation(self, env_state):
        pos = env_state
        obs = jnp.zeros((self.grid_size, self.grid_size, len(self.channels)), dtype=bool)
        obs = obs.at[pos[0],pos[1],self.channels['player']].set(True)
        # Flatten obs so we can input to a feed forward network, could skip this if you want to use a conv net
        return jnp.ravel(obs)

    def num_actions(self):
        return self._num_actions

# Wrapper for gymnax environments
class gymnax_env:
    def __init__(self, env_name):
        self.env, self.env_params = gymnax.make(env_name)

    @partial(jit, static_argnums=(0,))
    def step(self, key, env_state, action):
        obs, env_state, reward, terminal, aux = self.env.step(key, env_state, action, self.env_params)
        return env_state, obs, reward, terminal, aux

    @partial(jit, static_argnums=(0,))
    def reset(self, key):
        obs, env_state = self.env.reset(key, self.env_params)
        return env_state, obs

    @partial(jit, static_argnums=(0,))
    def get_observation(self, env_state):
        return self.env.get_obs(env_state)

    def num_actions(self):
        return self.env.num_actions

# Wrapper for gym environments, not jit compiled
class gym_env:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.obs = None

    # Note: key is not actually used here, but included for consistency (Gym only sets random key at reset)
    def step(self, key, env_state, action):
        # Truncated is not currently supported by this implementation (ignored)
        obs, reward, terminal, truncated, aux = env_state.step(action)
        self.obs = obs
        return env_state, jnp.asarray(obs, dtype=float), jnp.asarray(reward, dtype=float), jnp.asarray(terminal, dtype=bool), aux

    def reset(self, key):
        # Make a copy to return so the underlying env is not manipulated
        env_state = deepcopy(self.env)
        obs, _ = env_state.reset(seed=key[1])
        self.obs = obs
        return env_state, jnp.asarray(obs, dtype=float)

    def get_observation(self, env_state):
        return jnp.asarray(self.obs, dtype=float)

    def num_actions(self):
        return self.env.action_space.n
