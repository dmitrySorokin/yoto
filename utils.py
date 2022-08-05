import math
import torch
import gym
import numpy as np


class SpeedWrapper(gym.Wrapper):
    def __init__(self, e, target_vel):
        super().__init__(e)
        self.xpos = None
        self.target_vel = None
        self._max_episode_steps = 1000
        # self.observation_space = gym.spaces.Box(-np.inf, np.inf, (18,))

    def reset(self, v_target=None):
        obs = super().reset()
        self.xpos = self.env.sim.data.qpos[0]

        # target_vel \in (-2, 2)
        self.target_vel = v_target if v_target is not None else np.random.rand() * 4 - 2
        return np.concatenate([obs, [self.target_vel]])

    def step(self, action):
        obs, _, done, info = super().step(action)
        xpos, ypos, zpos = self.env.sim.data.qpos[:3]

        reward_ctrl = -0.1 * np.square(action).sum()
        velocity = (xpos - self.xpos) / self.dt

        info['velocity'] = velocity
        info['x_position'] = xpos
        info['y_position'] = ypos
        info['z_position'] = zpos
        info['v_target'] = self.target_vel

        reward_vel = max(1.0 - abs(1 - velocity / self.target_vel), 0.0) * 2

        reward = reward_ctrl + reward_vel
        done = done or zpos > 0.8
        obs = np.concatenate([obs, [self.target_vel]])

        self.xpos = xpos

        return obs, reward, done, info


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
