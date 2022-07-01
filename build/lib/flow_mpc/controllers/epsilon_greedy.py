import numpy as np

class EpsilonGreedyController:
    """
    Added by Shawn

    """
    def __init__(self, cost, lower_bound: list, upper_bound: list, epsilon=0.7, exploration_num=10, sample_num=50):
        """
        Initialization.
        :param cost: a callable function to measure the cost of a certain action in the given state
        :param lower_bound: lower bound of actions
        :param upper_bound: upper bound of actions
        :param epsilon: parameter related to exploration
        :param exploration_num: the lasting time of exploration mode
        :param sample_num: the number of different actions to sample
        """
        self.cost = cost
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.sample_num = sample_num
        self.epsilon = epsilon
        self.exploration_num = exploration_num
        self.exploration_count = 0      # the remaining number of exploration steps

    def step(self, step_data: tuple):
        """
        If the robot falls in crash with an obstacle and not in exploration mode, with probability of epsilon it'll enter exploration mode for a limited time
        While not in exploration mode, uniformly sample a bunch of actions and choose the best one.
        While in exploration mode, then uniformly sample an action
        :param state:
        :return:
        """
        if step_data.reward is not None and self.exploration_count == 0 and step_data.reward[0] and np.random.rand() > self.epsilon:
            self.exploration_count = self.exploration_num

        if self.exploration_count > 0:
            action = np.random.uniform(self.lower_bound, self.upper_bound)
            self.exploration_count -= 1
        else:
            candidates = np.random.uniform(self.lower_bound, self.upper_bound, (self.sample_num, len(self.lower_bound)))
            x = step_data.observation['robot_position'][0, 0:2]
            action = min(candidates, key=lambda u: self.cost(x, u))

        return action

    def reset(self):
        self.exploration_count = 0
