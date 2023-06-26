"""
Environment based on the description in the following paper:
https://arxiv.org/abs/1911.05815
"""

import numpy as np
import gymnasium as gym


class HighDimensionalDiabolicalCombinationLock(gym.Env):
    """Note: Hadamard rotation done as a part of preprocessor (not here)
    for speedup"""
    def __init__(self, horizon, noise_sigma,
                 n_actions=10, final_reward=10, seed=0, noise_seed=0):

        self.horizon = horizon
        self.noise_sigma = noise_sigma
        self.n_actions = n_actions
        self.final_reward = final_reward
        self.seed = seed
        self.noise_seed = noise_seed

        self.observation_dimension = int(np.power(2, np.ceil(np.log2(self.horizon + 4))))
        # self.rotation_matrix = hadamard(self.observation_dimension)

        self.random_state = np.random.RandomState(self.seed)
        self.noise_random_state = np.random.RandomState(self.noise_seed)
        self.optimal_actions = self.random_state.randint(n_actions,
                                                         size=(2,
                                                               self.horizon))

        self.row = None  # Either 0, 1, or 2
        self.column = None  # From 0 to horizon
        # self.row_representation = np.zeros(3)
        # self.col_representation = np.zeros(self.horizon + 1)

        self.action_space = gym.spaces.Discrete(n_actions)
        self.observation_space = gym.spaces.Box(low=-10.0, high=10.0,
                                                shape=(self.observation_dimension,),
                                                dtype=np.float32)
        self.render_mode = None

    def reset(self):
        info = {}
        self.row = self.noise_random_state.choice(2)
        self.column = 0
        s = self._generate_binary_state_vector(self.row, self.column)
        obs = self._generate_observation_vector_from_state(s)
        return obs, info

    def step(self, action):

        assert action < self.n_actions
        assert action >= 0

        reward = 0
        done = False
        info = {}
        # state transition

        # check if agent is not in the dead state
        if self.row != 2:
            if action == self.optimal_actions[self.row, self.column]:
                self.row = self.random_state.choice(2)
                reward = -1/(self.horizon-1)
            else:
                self.row = 2
                #reward = 0.1 * self.random_state.choice(2)

        self.column += 1
        # reward for final step
        if self.column == self.horizon:
            done = True
            if self.row == 2:
                reward = 0
            else:
                reward = self.final_reward

        s_prime = self._generate_binary_state_vector(self.row, self.column)
        o_prime = self._generate_observation_vector_from_state(s_prime)

        return o_prime, reward, done, False, info

    def _generate_binary_state_vector(self, row, column):
        row_representation = np.zeros(3)
        col_representation = np.zeros(self.horizon + 1)

        row_representation[row] = 1
        col_representation[column] = 1

        state_repsentation = np.concatenate((row_representation,
                                             col_representation))
        return state_repsentation

    def _generate_observation_vector_from_state(self, state):
        noise = self.noise_random_state.normal(scale=self.noise_sigma,
                                         size=state.shape)

        low_dim_obs = state + noise
        high_dim_obs = np.pad(low_dim_obs,
                              (0, self.observation_dimension - len(state)))
        # obs = np.matmul(self.rotation_matrix, high_dim_obs)
        # obs = high_dim_obs
        return high_dim_obs

    def state_extraction_key(self):
        return (self.row, self.column)
