import numpy as np
import matplotlib.pyplot as plt

class RandomController:
    """
    Added by Shawn
    A random controller that output a control sequence produced by Gaussian Process
    """
    def __init__(self, udim: int, urange: float, horizon: int, sigma: float=10, lower_bound: list=None, upper_bound: list=None):
        """
        Initialization.
        :param udim: the dimension of actions
        :param urange: the range of actions
        :param horizon: the control horizon
        :param sigma: the smoothness of actions
        :param lower_bound: lower bound of actions
        :param upper_bound: upper bound of actions
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.urange = urange
        self.udim = udim
        self.mean = np.zeros(horizon)
        self.cov = np.zeros((horizon, horizon))
        for i in range(horizon):
            for j in range(i, horizon):
                self.cov[i, j] = self.cov[j, i] = np.exp(-(i-j)**2/(2*sigma))   # Gaussian kernel

    def step(self, x):
        """
        Generate an action sequence. Not support batch yet
        """
        U = np.zeros((len(self.mean), self.udim))
        for i in range(self.udim):
            Utmp = self.urange * np.random.multivariate_normal(self.mean, self.cov)
            if self.lower_bound is not None or self.upper_bound is not None:
                Utmp = np.clip(Utmp, self.lower_bound[i], self.upper_bound[i])
            U[:, i] = Utmp
        return U

    def reset(self):
        return

if __name__ == "__main__":
    controller = RandomController(-3, 3, 40)
    for i in range(5):
        seq = controller.step(0)
        plt.plot(seq)
