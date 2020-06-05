import itertools
import random
import math

import torch
import torch.nn as nn
import torch.distributions.multivariate_normal as N
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np


from collections import namedtuple
from collections import OrderedDict
from itertools import product
from scipy.special import softmax

listToCheck = ['.running_mean', '.running_var', '.num_batches_tracked']

WeightEp = namedtuple('WeightEp', field_names=['weight', 'ave_reward', 'ave_steps', 'episodes'])
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action', 'action_probs'])
EpisodeStep_R = namedtuple('EpisodeStep', field_names=['observation', 'action', 'action_probs', 'reward'])
EpisodeStep_CEM = namedtuple('EpisodeStep', field_names=['observation', 'action'])

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

SEED = 110

class elite_ratio_buffer():
    def __init__(self, list_size):
        self.mylist = [np.NAN]*list_size
        self.mean = np.mean(self.mylist)
        # self.nanmean = np.nanmean(self.mylist)

    def update_buffer(self, ratio):
        self.mylist.pop(0)
        self.mylist.append(ratio)
        self.mean = np.mean(self.mylist)
        # self.mean = np.nanmean(self.mylist)


class elite_batch_bufferW_msCEM():
    def __init__(self, list_size):
        self.batch_list = [False]*list_size


    def update_buffer(self, elite_batch):
        self.batch_list.pop(0)
        self.batch_list.append(elite_batch)


    def get_single_batch(self):
        batch_buffer = []
        for batch in self.batch_list:
            try:
                batch_buffer.extend(batch)
            except TypeError:
                continue
        return batch_buffer


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def sample_episodesNorm(env, model, use_cuda, device):
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    # sm = nn.Softmax(dim=1)
    # print(use_cuda)
    # print(device)
    while True:
        obs_v = torch.FloatTensor([obs.flatten()])
        # print(obs_v)
        if use_cuda:
            obs_v = obs_v.to(device)
        # standardize observation vector
        obs_v_norm = (obs_v - obs_v.mean(dim=1, keepdim=True)) / obs_v.std(dim=1, keepdim=True)
        # print('obs_v:', obs_v.shape)
        # print(obs_v)
        # print(next(model.parameters()).is_cuda)
        act_probs_v = model(obs_v_norm)
        # act_probs_v = model.forward1(obs_v)
        # print(act_probs_v)
        act_probs = act_probs_v.cpu().data.numpy()[0]
        # print(act_probs)
        action = np.random.choice(len(act_probs), p=act_probs)
        # print(action)
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action, action_probs=act_probs))
        if is_done:
            return Episode(reward=episode_reward, steps=episode_steps)
        obs = next_obs


# def sample_episodesNorm_SingleLayer(env, model, seed, use_cuda, device):
#     episode_reward = 0.0
#     episode_steps = []
#     env.seed(seed)
#     obs = env.reset()
#
#     while True:
#         obs_v = torch.FloatTensor([obs.flatten()])
#         if use_cuda:
#             obs_v = obs_v.to(device)
#         # standardize observation vector
#         obs_v_norm = (obs_v - obs_v.mean(dim=1, keepdim=True)) / obs_v.std(dim=1, keepdim=True)
#         act_probs_v = model(obs_v_norm)
#         act_probs = act_probs_v.cpu().data.numpy()[0]
#         action = np.random.choice(len(act_probs), p=act_probs)
#         next_obs, reward, is_done, _ = env.step(action)
#         episode_reward += reward
#         episode_steps.append(EpisodeStep(observation=obs, action=action, action_probs=act_probs))
#         if is_done:
#             return Episode(reward=episode_reward, steps=episode_steps)
#         obs = next_obs


def sample_episodesNorm_eval(env, model, ensemble_size, seed, use_cuda, device):
    episode_reward = 0.0
    episode_steps = []
    env.seed(seed)
    obs = env.reset()
    while True:
        obs_v = torch.FloatTensor([obs.flatten()])
        if use_cuda:
            obs_v = obs_v.to(device)
        # standardize observation vector
        obs_v_norm = (obs_v - obs_v.mean(dim=1, keepdim=True)) / obs_v.std(dim=1, keepdim=True)

        output = model.ensemble_forward(obs_v_norm, ensemble_size)
        ensemble_out = torch.sum(output, dim=0).reshape(1, -1)
        act_probs_v = ensemble_out / torch.sum(ensemble_out, dim=1)

        act_probs = act_probs_v.cpu().data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action, action_probs=act_probs))
        if is_done:
            return Episode(reward=episode_reward, steps=episode_steps)
        obs = next_obs


def sample_episodes_eval_reacher(env, model, ensemble_size, max_steps, sparse_reward, screenshot, use_cuda, device):
    episode_reward = 0.0
    episode_steps = []

    obs = env.reset(screenshot)
    # sm = nn.Softmax(dim=1)
    for step in range(max_steps):
        obs_v = torch.FloatTensor([obs.flatten()])
        obs_v = obs_v.to(device)

        output = model.ensemble_forward(obs_v, ensemble_size)
        output[torch.isnan(output)] = 0
        ensemble_out = torch.mean(output, dim=0).reshape(1, -1)
        act_probs_v = ensemble_out
        action = act_probs_v.cpu().data.numpy().flatten()

        next_obs, reward, is_done, _ = env.step(action.flatten(), sparse_reward, screenshot)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action, action_probs=action))

        obs = next_obs

    return Episode(reward=episode_reward, steps=episode_steps)


def sample_episodesNorm_eval_Rstep(env, model, ensemble_size, seed, use_cuda, device):
    episode_reward = 0.0
    episode_steps = []
    env.seed(seed)
    obs = env.reset()
    # sm = nn.Softmax(dim=1)
    # print(use_cuda)
    # print(device)
    while True:
        obs_v = torch.FloatTensor([obs.flatten()])
        # print(obs_v)
        if use_cuda:
            obs_v = obs_v.to(device)
        # standardize observation vector
        obs_v_norm = (obs_v - obs_v.mean(dim=1, keepdim=True)) / obs_v.std(dim=1, keepdim=True)

        output = model.ensemble_forward(obs_v_norm, ensemble_size)
        ensemble_out = torch.sum(output, dim=0).reshape(1, -1)
        # act_probs_v = model.sm(ensemble_out)
        act_probs_v = ensemble_out / torch.sum(ensemble_out, dim=1)

        act_probs = act_probs_v.cpu().data.numpy()[0]
        # print(act_probs)
        action = np.random.choice(len(act_probs), p=act_probs)
        # print(action)
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep_R(observation=obs, action=action, action_probs=act_probs, reward=reward))
        if is_done:
            return Episode(reward=episode_reward, steps=episode_steps)
        obs = next_obs


def sample_episodesNorm_msCEM(env, model, seed=SEED):
    episode_reward = 0.0
    episode_steps = []
    env.seed(seed)  # set environment seed for not randomly generated maze
    obs = env.reset()
    # sm = nn.Softmax(dim=1)
    # print(use_cuda)
    # print(device)
    while True:
        obs_v = obs.flatten()
        # standardize observation vector
        obs_v_norm = (obs_v - np.mean(obs_v, axis=0)) / np.std(obs_v, axis=0)
        act_probs = model.forward(obs_v_norm)
        act_probs = softmax(act_probs)
        action = np.random.choice(len(act_probs.flatten()), p=act_probs.flatten())
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action, action_probs=act_probs))
        if is_done:
            return Episode(reward=episode_reward, steps=episode_steps)
        obs = next_obs


def sample_episodesNorm_msCEM_genPerStep(env, model, theta_mean, theta_std, seed=SEED):
    episode_reward = 0.0
    episode_steps = []
    env.seed(seed)  # set environment seed for not randomly generated maze
    obs = env.reset()

    theta_list = []
    while True:
        # keep generating new theta at each ep step
        theta = np.random.multivariate_normal(theta_mean, np.diag(theta_std ** 2))
        theta_list.append(theta)
        model.set_theta(theta)

        obs_v = obs.flatten()
        # standardize observation vector
        obs_v_norm = (obs_v - np.mean(obs_v, axis=0)) / np.std(obs_v, axis=0)
        act_probs = model.forward(obs_v_norm)
        act_probs = softmax(act_probs)
        action = np.random.choice(len(act_probs.flatten()), p=act_probs.flatten())
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action, action_probs=act_probs))
        if is_done:
            return Episode(reward=episode_reward, steps=episode_steps), theta_list
        obs = next_obs


def sample_episodes_netCEM(env, model, use_cuda, device):
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax(dim=1)
    # print(use_cuda)
    # print(device)
    while True:
        obs_v = torch.FloatTensor([obs.flatten()])
        # print(obs_v)
        if use_cuda:
            obs_v = obs_v.to(device)
        # standardize observation vector
        obs_v_norm = (obs_v - obs_v.mean(dim=1, keepdim=True)) / obs_v.std(dim=1, keepdim=True)
        # print('obs_v:', obs_v.shape)
        # print(obs_v)
        # print(next(model.parameters()).is_cuda)
        act_probs_v = sm(model(obs_v_norm))
        # act_probs_v = model(obs_v)
        # act_probs_v = model.forward1(obs_v)
        # print(act_probs_v)
        act_probs = act_probs_v.cpu().data.numpy()[0]
        # print(act_probs)
        action = np.random.choice(len(act_probs), p=act_probs)
        # print(action)
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action, action_probs=act_probs))
        if is_done:
            return Episode(reward=episode_reward, steps=episode_steps)
        obs = next_obs


def sample_episodes_netCEM2(env, model, use_cuda, device, seed=SEED):
    episode_reward = 0.0
    episode_steps = []
    env.seed(seed)  # set environment seed for not randomly generated maze
    obs = env.reset()
    sm = nn.Softmax(dim=1)
    while True:
        obs_v = torch.FloatTensor([obs.flatten()])
        if use_cuda:
            obs_v = obs_v.to(device)
        # standardize observation vector
        obs_v_norm = (obs_v - obs_v.mean(dim=1, keepdim=True)) / obs_v.std(dim=1, keepdim=True)
        # act_probs_v = sm(model(obs_v_norm))
        # act_probs = act_probs_v.cpu().data.numpy()[0]
        # act_probs = np.nan_to_num(act_probs)
        # act_probs = softmax(act_probs)
        model_out = model(obs_v_norm)
        model_out[torch.isnan(model_out)] = 0
        act_probs_v = sm(model_out)
        act_probs = act_probs_v.cpu().data.numpy()[0]

        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action, action_probs=act_probs))
        if is_done:
            return Episode(reward=episode_reward, steps=episode_steps)
        obs = next_obs


def sample_episodes_netCEM2_reacher(env, model, max_steps, sparse_reward, screenshot, use_cuda, device):
    episode_reward = 0.0
    episode_steps = []

    obs = env.reset(screenshot)
    for step in range(max_steps):
        obs_v = torch.FloatTensor([obs.flatten()])
        obs_v = obs_v.to(device)

        model_out = model(obs_v)
        model_out[torch.isnan(model_out)] = 0
        act_probs_v = model_out
        action = act_probs_v.cpu().data.numpy().flatten()

        next_obs, reward, is_done, _ = env.step(action.flatten(), sparse_reward, screenshot)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action, action_probs=action))

        obs = next_obs

    return Episode(reward=episode_reward, steps=episode_steps)


def sample_episodes_net_Rstep(env, model, use_cuda, device, seed=SEED):
    episode_reward = 0.0
    episode_steps = []
    env.seed(seed)  # set environment seed for not randomly generated maze
    obs = env.reset()
    sm = nn.Softmax(dim=1)
    # print(use_cuda)
    # print(device)
    while True:
        obs_v = torch.FloatTensor([obs.flatten()])
        if use_cuda:
            obs_v = obs_v.to(device)
        # standardize observation vector
        obs_v_norm = (obs_v - obs_v.mean(dim=1, keepdim=True)) / (obs_v.std(dim=1, keepdim=True) + 1e-9)
        act_probs_v = sm(model(obs_v_norm))
        # act_probs_v = model(obs_v)
        # act_probs_v = model.forward1(obs_v)
        # print(act_probs_v)
        act_probs = act_probs_v.cpu().data.numpy()[0]
        # print(act_probs)
        action = np.random.choice(len(act_probs), p=act_probs)
        # print(action)
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep_R(observation=obs, action=action, action_probs=act_probs, reward=reward))
        if is_done:
            return Episode(reward=episode_reward, steps=episode_steps)
        obs = next_obs


def sample_episodes_net_Rstep_DDPG(env, model, memory, use_cuda, device, seed=SEED, train=True):
    episode_reward = 0.0
    episode_steps = []
    env.seed(seed)  # set environment seed for not randomly generated maze
    obs = env.reset()
    # sm = nn.Softmax(dim=1)
    while True:
        state = np.float32(obs.flatten())
        # standardize state vector
        state_norm = (state - np.mean(state, axis=0)) / np.std(state, axis=0)
        # select action
        if train == True:
            act_output = model.get_exploration_action(state_norm)
            act_probs = softmax(act_output)
            action = np.random.choice(len(act_probs), p=act_probs)
        else:
            act_output = model.get_exploitation_action(state_norm)
            act_probs = softmax(act_output)
            action = np.random.choice(len(act_probs), p=act_probs)

        next_obs, reward, is_done, info = env.step(action)
        episode_reward += reward


        next_state = np.float32(next_obs.flatten())
        # push this exp in ram
        if train == True:
            memory.add(state, act_probs, reward, next_state)

        episode_steps.append(
            EpisodeStep_R(observation=state, action=action, action_probs=act_probs, reward=reward))
        if is_done:
            return Episode(reward=episode_reward, steps=episode_steps)
        obs = next_obs


def sample_episodes_net_Rstep_reacher(env, model, n_actions, steps_done, eps_start, eps_end, eps_decay, max_steps, sparse_reward, screenshot, use_cuda, device):
    episode_reward = 0.0
    episode_steps = []

    obs = env.reset(screenshot)
    sm = nn.Softmax(dim=1)
    for step in range(max_steps):
        obs_v = torch.FloatTensor([obs.flatten()])
        obs_v = obs_v.to(device)
        model_out = model(obs_v)
        # model_out[torch.isnan(model_out)] = 0  # do not use this because need calculate gradient
        act_probs_v = model_out
        # action = act_probs_v.cpu().data.numpy().flatten()

        # select action
        sample = random.random()
        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
        steps_done += 1
        if sample > eps_threshold:
            # action = model(obs_v_norm).max(1)[1].view(1, 1)
            action = act_probs_v.flatten()
        else:
            # action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
            # action = torch.tensor(np.random.uniform(-1.0, 1.0, size=(n_actions)), device=device)  # random action
            action = act_probs_v.flatten()  # still uses the policy action

        next_obs, reward, is_done, _ = env.step(action, sparse_reward, screenshot)
        episode_reward += reward
        # reward_v = torch.tensor([reward], device=device)
        # next_obs_v = torch.FloatTensor([next_obs.flatten()])
        # next_obs_v = next_obs_v.to(device)

        episode_steps.append(EpisodeStep_R(observation=obs, action=action.cpu().data.numpy(), action_probs=action.cpu().data.numpy(), reward=reward))

        obs = next_obs

    return Episode(reward=episode_reward, steps=episode_steps)


def sample_episodes_net_Rstep_reacher_DDPG(env, model, memory, n_actions, max_steps, sparse_reward, screenshot, use_cuda, device, train=True):
    episode_reward = 0.0
    episode_steps = []

    obs = env.reset(screenshot)
    for step in range(max_steps):
        state = np.float32(obs)
        # select action
        if train==True:
            action = model.get_exploration_action(state)
        else:
            action = model.get_exploitation_action(state)

        next_obs, reward, is_done, info = env.step(action, sparse_reward, screenshot)
        episode_reward += reward

        if is_done:
            new_state = None
        else:
            next_state = np.float32(next_obs)
            # push this exp in ram
            if train == True:
                memory.add(state, action, reward, next_state)

        episode_steps.append(
            EpisodeStep_R(observation=state, action=action, action_probs=action, reward=reward))
        obs = next_obs

    return Episode(reward=episode_reward, steps=episode_steps)


def sample_episodes_net_Rstep_DQN(env, model, memory, n_actions, steps_done, eps_start, eps_end, eps_decay, use_cuda, device, seed=SEED):
    episode_reward = 0.0
    episode_steps = []
    env.seed(seed)  # set environment seed for not randomly generated maze
    obs = env.reset()
    sm = nn.Softmax(dim=1)

    obs_v = torch.FloatTensor([obs.flatten()])
    obs_v = obs_v.to(device)
    # standardize observation vector
    obs_v_norm = (obs_v - obs_v.mean(dim=1, keepdim=True)) / obs_v.std(dim=1, keepdim=True)

    while True:
        # get action probabilities, but is not used since DQN here not stochastic policies
        act_probs_v = sm(model(obs_v_norm))
        act_probs = act_probs_v.cpu().data.numpy()[0]

        # select action
        sample = random.random()
        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
        steps_done += 1  # enabled steps count +1 per step in each episode
        if sample > eps_threshold:
            action = model(obs_v_norm).max(1)[1].view(1, 1)
        else:
            action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

        next_obs, reward, is_done, _ = env.step(action.item())
        episode_reward += reward
        reward_v = torch.tensor([reward], device=device)
        next_obs_v = torch.FloatTensor([next_obs.flatten()])
        next_obs_v = next_obs_v.to(device)
        # standardize observation vector
        next_obs_v_norm = (next_obs_v - next_obs_v.mean(dim=1, keepdim=True)) / next_obs_v.std(dim=1, keepdim=True)

        # Store the transition in memory
        memory.push(obs_v_norm, action, next_obs_v_norm, reward_v)

        # Move to the next state
        obs_v_norm = next_obs_v_norm

        episode_steps.append(EpisodeStep_R(observation=obs, action=action, action_probs=act_probs, reward=reward))
        if is_done:
            return memory, steps_done, Episode(reward=episode_reward, steps=episode_steps)
        obs = next_obs


def sample_episodes_net_Rstep_DQN_reacher(env, model, memory, n_actions, steps_done, eps_start, eps_end, eps_decay, max_steps, sparse_reward, screenshot, use_cuda, device):
    episode_reward = 0.0
    episode_steps = []

    obs = env.reset(screenshot)
    # sm = nn.Softmax(dim=1)
    obs_v = torch.FloatTensor([obs.flatten()])
    obs_v = obs_v.to(device)

    for step in range(max_steps):
        # get action probabilities, but is not used since DQN here not stochastic policies
        model_out = model(obs_v)
        # model_out[torch.isnan(model_out)] = 0  # do not use this because need calculate gradient
        act_probs_v = model_out
        # action = act_probs_v.cpu().data.numpy().flatten()

        # select action
        sample = random.random()
        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
        steps_done += 1
        if sample > eps_threshold:
            # action = model(obs_v_norm).max(1)[1].view(1, 1)
            action = act_probs_v.flatten()
        else:
            # action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
            action = torch.tensor(np.random.uniform(-1.0, 1.0, size=(n_actions)), device=device)

        next_obs, reward, is_done, _ = env.step(action, sparse_reward, screenshot)
        episode_reward += reward
        reward_v = torch.tensor([reward], device=device)
        next_obs_v = torch.FloatTensor([next_obs.flatten()])
        next_obs_v = next_obs_v.to(device)

        # Store the transition in memory
        memory.push(obs_v, action, next_obs_v, reward_v)

        # Move to the next state
        obs_v = next_obs_v

        episode_steps.append(EpisodeStep_R(observation=obs, action=action, action_probs=action, reward=reward))

        obs = next_obs

    return memory, steps_done, Episode(reward=episode_reward, steps=episode_steps)



def sample_episodes2(env, model, use_cuda, device):
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax(dim=1)
    # print(use_cuda)
    # print(device)
    while True:
        obs_v = torch.FloatTensor([obs.flatten()])
        # print(obs_v)
        if use_cuda:
            obs_v = obs_v.to(device)
        # print('obs_v:', obs_v.shape)
        # print(obs_v)
        # print(next(model.parameters()).is_cuda)
        act_probs_v = sm(model(obs_v))
        # act_probs_v = model(obs_v)
        # act_probs_v = model.forward1(obs_v)
        # print(act_probs_v)
        act_probs = act_probs_v.cpu().data.numpy()[0]
        # print(act_probs)
        action = np.random.choice(len(act_probs), p=act_probs)
        # print(action)
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action, action_probs=act_probs))
        if is_done:
            return Episode(reward=episode_reward, steps=episode_steps)
        obs = next_obs


def sample_episodeW(env, model, use_cuda, device):
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    # sm = nn.Softmax(dim=1)
    # print(use_cuda)
    # print(device)
    while True:
        obs_v = torch.FloatTensor([obs.flatten()])
        # print(obs_v)
        if use_cuda:
            obs_v = obs_v.to(device)
        # print('obs_v:', obs_v.shape)
        # print(obs_v)
        # print(next(model.parameters()).is_cuda)
        act_probs_v = model(obs_v)
        # act_probs_v = model.forward1(obs_v)
        # print(act_probs_v)
        act_probs = act_probs_v.cpu().data.numpy()[0]
        # print(act_probs)
        action = np.random.choice(len(act_probs), p=act_probs)
        # print(action)
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action, action_probs=act_probs))
        if is_done:
            return Episode(reward=episode_reward, steps=episode_steps)
        obs = next_obs


# def iterate_model_batchesNorm_msCEM(env, model, batch_size):
#     batch = []
#     for episodes in range(batch_size):
#         batch.append(sample_episodesNorm_msCEM(env, model))
#     return batch


# def iterate_model_batches_net_Rstep(env, model, batch_size, use_cuda=False, device=torch.device("cuda:0")):
#     batch = []
#     for episodes in range(batch_size):
#         batch.append(sample_episodes_net_Rstep(env, model, use_cuda, device))
#     return batch


def iterate_model_batches_net_Rstep_DDPG(env, model, memory, batch_size, use_cuda=False, device=torch.device("cuda:0"), train=False):
    batch = []
    for episodes in range(batch_size):
        batch.append(sample_episodes_net_Rstep_DDPG(env, model, memory, use_cuda, device, train))
    return batch


# def iterate_model_batches_net_Rstep_reacher(env, model, batch_size, n_actions, steps_done, eps_start, eps_end, eps_decay, max_steps, sparse_reward, screenshot, use_cuda=False, device=torch.device("cuda:0")):
#     batch = []
#     for episodes in range(batch_size):
#         batch.append(sample_episodes_net_Rstep_reacher(env, model, n_actions, steps_done, eps_start, eps_end, eps_decay, max_steps, sparse_reward, screenshot, use_cuda, device))
#     return batch
#
#
# def iterate_model_batches_net_Rstep_reacher_DDPG(env, model, memory, batch_size, n_actions, max_steps, sparse_reward, screenshot, use_cuda=False, device=torch.device("cuda:0"), train=True):
#     batch = []
#     for episodes in range(batch_size):
#         batch.append(sample_episodes_net_Rstep_reacher_DDPG(env, model, memory, n_actions, max_steps, sparse_reward, screenshot, use_cuda, device, train))
#     return batch


# def iterate_model_batches_net_Rstep_DQN(env, model, memory, n_actions, steps_done, eps_start, eps_end, eps_decay, batch_size, use_cuda=False, device=torch.device("cuda:0"), seed=SEED):
#     batch = []
#     steps_done_ep = steps_done
#     for episodes in range(batch_size):
#         memory, steps_done_ep, sampled_ep = sample_episodes_net_Rstep_DQN(env, model, memory, n_actions, steps_done_ep,
#                                                                              eps_start, eps_end, eps_decay, use_cuda,
#                                                                              device, seed)
#         batch.append(sampled_ep)
#         # steps_done += 1
#     return memory, steps_done_ep, batch


# def iterate_model_batches_netCEM(env, model, batch_size, use_cuda=False, device=torch.device("cuda:0")):
#     batch = []
#     for episodes in range(batch_size):
#         batch.append(sample_episodes_netCEM(env, model, use_cuda, device))
#     return batch


# def iterate_model_batchesWB_msCEM(env, model, theta, agent_start_list, batch_size, seed):
#     batch = []
#     # set the current model's generated weights
#     model.set_theta(theta)
#     # sample episodes
#     for agent_start in agent_start_list:
#         start_pos = agent_start[0]  # col, row
#         start_dir = agent_start[1]
#         env.set_initial_pos_dir(start_pos, start_dir)
#         ep_batch = []
#         for episodes in range(batch_size):
#             ep_batch.append(sample_episodesNorm_msCEM(env, model, seed=seed))
#         batch.extend(ep_batch)
#         # print(len(batch))
#     # also calculate the average reward of this model batch
#     rewards = list(map(lambda s: s.reward, batch))
#     reward_mean = float(np.mean(rewards))
#     step_counts = list(map(lambda s: len(s.steps), batch))
#     step_counts_mean = float(np.mean(step_counts))
#
#     return WeightEp(weight=theta, ave_reward=reward_mean, ave_steps=step_counts_mean, episodes=batch)


def iterate_model_batchesWB_msCEM2(env, model, theta, agent_start_list, batch_size, use_cuda, device, seed):
    batch = []
    # set the current model's generated weights
    # model.set_theta(theta)
    theta_tensor = torch.FloatTensor(theta.flatten())
    loadWeight_dict = getLoadWeightDict(model.state_dict(), theta_tensor)
    model.load_state_dict(loadWeight_dict)

    # sample episodes
    for agent_start in agent_start_list:
        start_pos = agent_start[0]  # col, row
        start_dir = agent_start[1]
        env.set_initial_pos_dir(start_pos, start_dir)
        ep_batch = []
        for episodes in range(batch_size):
            ep_batch.append(sample_episodes_netCEM2(env, model, use_cuda, device, seed=seed))
        batch.extend(ep_batch)
        # print(len(batch))
    # also calculate the average reward of this model batch
    rewards = list(map(lambda s: s.reward, batch))
    reward_mean = float(np.mean(rewards))
    step_counts = list(map(lambda s: len(s.steps), batch))
    step_counts_mean = float(np.mean(step_counts))

    return WeightEp(weight=theta, ave_reward=reward_mean, ave_steps=step_counts_mean, episodes=batch)


def iterate_model_batchesWB_msCEM2_reacher(env, model, theta, agent_start_list, max_steps, batch_size, sparse_reward, screenshot, use_cuda, device):
    batch = []
    # set the current model's generated weights
    theta_tensor = torch.FloatTensor(theta.flatten())
    loadWeight_dict = getLoadWeightDict(model.state_dict(), theta_tensor)
    model.load_state_dict(loadWeight_dict)

    # sample episodes
    for agent_start in agent_start_list:
        ep_batch = []
        for episodes in range(batch_size):
            ep_batch.append(sample_episodes_netCEM2_reacher(env, model, max_steps, sparse_reward, screenshot, use_cuda, device))
        batch.extend(ep_batch)
    # also calculate the average reward of this model batch
    rewards = list(map(lambda s: s.reward, batch))
    reward_mean = float(np.mean(rewards))
    step_counts = list(map(lambda s: len(s.steps), batch))
    step_counts_mean = float(np.mean(step_counts))

    return WeightEp(weight=theta, ave_reward=reward_mean, ave_steps=step_counts_mean, episodes=batch)


# def filter_batch(batch, percentile):
#     rewards = list(map(lambda s: s.reward, batch))
#     reward_bound = np.percentile(rewards, percentile)
#     reward_mean = float(np.mean(rewards))
#     step_counts = list(map(lambda s: len(s.steps), batch))
#     step_counts_mean = float(np.mean(step_counts))
#
#     train_obs = []
#     train_act = []
#     train_act_probs = []
#     unfit_count = 0
#     for example in batch:
#         if example.reward < reward_bound:
#             unfit_count += 1
#             continue
#         train_obs.extend(map(lambda step: step.observation, example.steps))
#         train_act.extend(map(lambda step: step.action, example.steps))
#         train_act_probs.extend(map(lambda step: step.action_probs, example.steps))
#
#     train_obs_v = torch.FloatTensor(train_obs)
#     train_act_v = torch.LongTensor(train_act)  # train_act needs to be a list
#     train_act_probs = torch.FloatTensor(train_act_probs)
#     return train_obs_v, train_act_v, train_act_probs, reward_bound, reward_mean, np.max(rewards), np.min(step_counts), step_counts_mean, unfit_count


# def filter_elite_batch(batch, percentile):
#     rewards = list(map(lambda s: s.reward, batch))
#     reward_bound = np.percentile(rewards, percentile)
#     reward_mean = float(np.mean(rewards))
#     reward_max = float(np.max(rewards))
#     step_counts = list(map(lambda s: len(s.steps), batch))
#     step_counts_mean = float(np.mean(step_counts))
#
#     elite_batch = []
#     top_batch =[]
#     train_obs = []
#     train_act = []
#     train_act_probs = []
#     unfit_count = 0
#     for example in batch:
#         if example.reward < reward_bound:
#             unfit_count += 1
#             continue
#         elite_batch.append(example)
#         if example.reward == reward_max:
#             top_batch.append(example)
#         train_obs.extend(map(lambda step: step.observation, example.steps))
#         train_act.extend(map(lambda step: step.action, example.steps))
#         train_act_probs.extend(map(lambda step: step.action_probs, example.steps))
#
#     train_obs_v = torch.FloatTensor(train_obs)
#     # train_act_v = torch.LongTensor(train_act)
#     train_act_v = torch.FloatTensor(train_act)
#     train_act_probs = torch.FloatTensor(train_act_probs)
#     return elite_batch, top_batch, train_obs_v, train_act_v, train_act_probs, reward_bound, reward_mean, reward_max, np.min(step_counts), step_counts_mean, unfit_count


# def filter_elite_batch_accumR(batch, percentile):
#     rewards = list(map(lambda s: s.reward, batch))
#     reward_bound = np.percentile(rewards, percentile)
#     reward_mean = float(np.mean(rewards))
#     reward_max = float(np.max(rewards))
#     reward_accum = float(np.sum(rewards))
#     step_counts = list(map(lambda s: len(s.steps), batch))
#     step_counts_mean = float(np.mean(step_counts))
#
#     elite_batch = []
#     top_batch =[]
#     train_obs = []
#     train_act = []
#     train_act_probs = []
#     unfit_count = 0
#     for example in batch:
#         if example.reward < reward_bound:
#             unfit_count += 1
#             continue
#         elite_batch.append(example)
#         if example.reward == reward_max:
#             top_batch.append(example)
#         train_obs.extend(map(lambda step: step.observation, example.steps))
#         train_act.extend(map(lambda step: step.action, example.steps))
#         train_act_probs.extend(map(lambda step: step.action_probs, example.steps))
#
#     train_obs_v = torch.FloatTensor(train_obs)
#     # train_act_v = torch.LongTensor(train_act)
#     train_act_v = torch.FloatTensor(train_act)
#     train_act_probs = torch.FloatTensor(train_act_probs)
#     return elite_batch, top_batch, train_obs_v, train_act_v, train_act_probs, reward_bound, reward_mean, reward_max, reward_accum, np.min(step_counts), step_counts_mean, unfit_count


# def filter_elite_batch2(batch, percentile):
#     rewards = list(map(lambda s: s.reward, batch))
#     up_reward_bound = np.percentile(rewards, percentile)
#     low_reward_bound = np.percentile(rewards, 100-percentile)
#     reward_mean = float(np.mean(rewards))
#     reward_max = float(np.max(rewards))
#     step_counts = list(map(lambda s: len(s.steps), batch))
#     step_counts_mean = float(np.mean(step_counts))
#
#     elite_batch = []
#     top_batch =[]
#     bot_batch = []
#     train_obs = []
#     train_act = []
#     train_act_probs = []
#     unfit_count = 0
#     for example in batch:
#         if example.reward <= low_reward_bound:
#             bot_batch.append(example)
#         if example.reward < up_reward_bound:
#             unfit_count += 1
#             continue
#         elite_batch.append(example)
#         if example.reward == reward_max:
#             top_batch.append(example)
#         train_obs.extend(map(lambda step: step.observation, example.steps))
#         train_act.extend(map(lambda step: step.action, example.steps))
#         train_act_probs.extend(map(lambda step: step.action_probs, example.steps))
#
#     train_obs_v = torch.FloatTensor(train_obs)
#     # train_act_v = torch.LongTensor(train_act)
#     train_act_v = torch.FloatTensor(train_act)
#     train_act_probs = torch.FloatTensor(train_act_probs)
#     return elite_batch, top_batch, bot_batch, train_obs_v, train_act_v, train_act_probs, up_reward_bound, reward_mean, reward_max, np.min(step_counts), step_counts_mean, unfit_count


def concatWeights(weightList, device):
    weight_tensor = torch.zeros(0).to(device)
    for weight in weightList:
        weight_tensor = torch.cat([weight_tensor, weight.flatten()], dim=0)
    return weight_tensor.reshape(1, -1)


def concatWeights_3D(weightList, device):
    mod_weightList = []
    for weight in weightList:
        mod_weightList.append(weight.reshape(weight.shape[0],-1))
    weight_tensor = torch.cat(mod_weightList, dim=1)
    return weight_tensor


# def filter_elite_batchW(batch, percentile, device):
#     # take note the rewards here are already the average rewards of the individual models (each model has a set of episodes)
#     rewards = list(map(lambda s: s.ave_reward, batch))
#     reward_bound = np.percentile(rewards, percentile)
#     reward_mean = float(np.mean(rewards))
#     reward_max = float(np.max(rewards))
#     step_counts = list(map(lambda s: s.ave_steps, batch))
#     step_counts_mean = float(np.mean(step_counts))
#
#     elite_batch = []
#     top_batch =[]
#     train_weights = []
#     unfit_count = 0
#     for example in batch:
#         if example.ave_reward < reward_bound:
#             unfit_count += 1
#             continue
#         elite_batch.append(example)
#         if example.ave_reward == reward_max:
#             top_batch.append(example)
#
#         train_weights.append(concatWeights(example.weight, device))
#
#     train_weights_v = torch.cat(train_weights, dim=0)
#     return elite_batch, top_batch, train_weights_v, reward_bound, reward_mean, reward_max, np.min(step_counts), step_counts_mean, unfit_count
#
#
# def filter_elite_batchW_msCEM(batch, percentile):
#     # take note the rewards here are already the average rewards of the individual models (each model has a set of episodes)
#     rewards = list(map(lambda s: s.ave_reward, batch))
#     reward_bound = np.percentile(rewards, percentile)
#     reward_mean = float(np.mean(rewards))
#     reward_max = float(np.max(rewards))
#     step_counts = list(map(lambda s: s.ave_steps, batch))
#     step_counts_mean = float(np.mean(step_counts))
#
#     elite_batch = []
#     top_batch =[]
#     elite_weights = []
#     unfit_count = 0
#     for example in batch:
#         if example.ave_reward < reward_bound:
#             unfit_count += 1
#             continue
#         elite_batch.append(example)
#         if example.ave_reward == reward_max:
#             top_batch.append(example)
#
#         elite_weights.append(example.weight)
#
#     # train_weights_v = torch.cat(train_weights, dim=0)
#     return elite_batch, top_batch, elite_weights, reward_bound, reward_mean, reward_max, np.min(step_counts), step_counts_mean, unfit_count
#
#
# def filter_elite_batchW_msCEM_accumR(batch, percentile):
#     # take note the rewards here are already the average rewards of the individual models (each model has a set of episodes)
#     rewards = list(map(lambda s: s.ave_reward, batch))
#     reward_bound = np.percentile(rewards, percentile)
#     reward_mean = float(np.mean(rewards))
#     reward_max = float(np.max(rewards))
#     reward_accum = float(np.sum(rewards))
#     step_counts = list(map(lambda s: s.ave_steps, batch))
#     step_counts_mean = float(np.mean(step_counts))
#
#     elite_batch = []
#     top_batch =[]
#     elite_weights = []
#     unfit_count = 0
#     for example in batch:
#         if example.ave_reward < reward_bound:
#             unfit_count += 1
#             continue
#         elite_batch.append(example)
#         if example.ave_reward == reward_max:
#             top_batch.append(example)
#
#         elite_weights.append(example.weight)
#
#     # train_weights_v = torch.cat(train_weights, dim=0)
#     return elite_batch, top_batch, elite_weights, reward_bound, reward_mean, reward_max, reward_accum, np.min(step_counts), step_counts_mean, unfit_count


def filter_elite_batchW_msCEM_accumR_std(batch, percentile):
    # take note the rewards here are already the average rewards of the individual models (each model has a set of episodes)
    rewards = list(map(lambda s: s.ave_reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))
    reward_std = float(np.std(rewards))
    reward_max = float(np.max(rewards))
    reward_accum = float(np.sum(rewards))
    step_counts = list(map(lambda s: s.ave_steps, batch))
    step_counts_mean = float(np.mean(step_counts))

    elite_batch = []
    top_batch =[]
    elite_weights = []
    unfit_count = 0
    for example in batch:
        if example.ave_reward < reward_bound:
            unfit_count += 1
            continue
        elite_batch.append(example)
        if example.ave_reward == reward_max:
            top_batch.append(example)

        elite_weights.append(example.weight)

    # train_weights_v = torch.cat(train_weights, dim=0)
    return elite_batch, top_batch, elite_weights, reward_bound, reward_mean, reward_std, reward_max, reward_accum, np.min(step_counts), step_counts_mean, unfit_count


def convert_obs_act(batch):
    train_obs = []
    train_act = []
    train_act_probs = []
    for example in batch:
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))
        train_act_probs.extend(map(lambda step: step.action_probs, example.steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.FloatTensor(train_act)
    train_act_probs = torch.FloatTensor(train_act_probs)
    return train_obs_v, train_act_v, train_act_probs


def convert_weight_tensors(batch, device):
    train_weights = []
    for example in batch:
        train_weights.append(concatWeights(example.weight, device))

    train_weights_v = torch.cat(train_weights, dim=0)
    return train_weights_v


def convert_weight_thetas(batch):
    train_weights = []
    for example in batch:
        train_weights.append(example.weight)

    return train_weights


def concat_weight_thetas(batch):
    train_weights = []
    for example in batch:
        train_weights.extend(example.weight)

    return train_weights


def convert_weightbatch_obs_act(batch):
    train_obs = []
    train_act = []
    train_act_probs = []
    for net in batch:
        for ep in net.episodes:
            train_obs.extend(map(lambda step: step.observation, ep.steps))
            train_act.extend(map(lambda step: step.action, ep.steps))
            train_act_probs.extend(map(lambda step: step.action_probs, ep.steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.FloatTensor(train_act)
    train_act_probs = torch.FloatTensor(train_act_probs)
    return train_obs_v, train_act_v, train_act_probs


# def get_batch_stats(batch):
#     rewards = list(map(lambda s: s.reward, batch))
#     reward_mean = float(np.mean(rewards))
#     step_counts = list(map(lambda s: len(s.steps), batch))
#     step_counts_mean = float(np.mean(step_counts))
#     return reward_mean, step_counts_mean


# def get_batch_stats_accumR(batch):
#     rewards = list(map(lambda s: s.reward, batch))
#     reward_mean = float(np.mean(rewards))
#     reward_accum = float(np.sum(rewards))
#     step_counts = list(map(lambda s: len(s.steps), batch))
#     step_counts_mean = float(np.mean(step_counts))
#     return reward_mean, reward_accum, step_counts_mean


def get_batch_stats_eval(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_mean = float(np.mean(rewards))
    reward_max = float(np.max(rewards))
    reward_accum = float(np.sum(rewards))
    reward_std = float(np.std(rewards))
    reward_bound = np.percentile(rewards, percentile)
    step_counts = list(map(lambda s: len(s.steps), batch))
    step_counts_mean = float(np.mean(step_counts))
    step_counts_std = float(np.std(step_counts))

    elite_count = 0
    for example in batch:
        if example.reward >= reward_bound:
            elite_count += 1

    elite_ratio = elite_count / len(batch)

    return reward_mean, reward_std, reward_max, reward_accum, step_counts_mean, step_counts_std, elite_ratio


def get_batch_statsW(batch):
    rewards = list(map(lambda s: s.ave_reward, batch))
    reward_mean = float(np.mean(rewards))
    step_counts = list(map(lambda s: s.ave_steps, batch))
    step_counts_mean = float(np.mean(step_counts))
    return reward_mean, step_counts_mean


# def get_batch_accumR(batch, gamma=0.9):
#     rewards = list(map(lambda s: s.reward, batch))
#     reward_mean = float(np.mean(rewards))
#     reward_max = float(np.max(rewards))
#     reward_accum = float(np.sum(rewards))
#     step_counts = list(map(lambda s: len(s.steps), batch))
#     step_counts_mean = float(np.mean(step_counts))
#
#     train_obs = []
#     train_act = []
#     train_act_probs = []
#     train_rewards = []
#     discount_rewards = []
#
#     for example in batch:
#         train_obs.extend(map(lambda step: step.observation, example.steps))
#         train_act.extend(map(lambda step: step.action, example.steps))
#         train_act_probs.extend(map(lambda step: step.action_probs, example.steps))
#
#         reward_list = list(map(lambda step: step.reward, example.steps))
#         discounted_reward = get_discounted_rewards(reward_list, gamma)
#         train_rewards.extend(reward_list)
#         discount_rewards.extend(discounted_reward)
#
#     train_obs_v = torch.FloatTensor(train_obs)
#     train_act_v = torch.LongTensor(train_act)
#     train_act_probs = torch.FloatTensor(train_act_probs)
#     train_rewards = torch.FloatTensor(train_rewards)
#     train_discount_rewards = torch.FloatTensor(discount_rewards)
#     return train_obs_v, train_act_v, train_act_probs, train_rewards, train_discount_rewards, reward_mean, reward_max, reward_accum, np.min(step_counts), step_counts_mean


def get_batch_accumR_std(batch, gamma=0.9):
    rewards = list(map(lambda s: s.reward, batch))
    reward_mean = float(np.mean(rewards))
    reward_max = float(np.max(rewards))
    reward_accum = float(np.sum(rewards))
    reward_std = float(np.std(rewards))
    step_counts = list(map(lambda s: len(s.steps), batch))
    step_counts_mean = float(np.mean(step_counts))

    train_obs = []
    train_act = []
    train_act_probs = []
    train_rewards = []
    discount_rewards = []

    for example in batch:
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))
        train_act_probs.extend(map(lambda step: step.action_probs, example.steps))

        reward_list = list(map(lambda step: step.reward, example.steps))
        discounted_reward = get_discounted_rewards(reward_list, gamma)
        train_rewards.extend(reward_list)
        discount_rewards.extend(discounted_reward)

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    train_act_probs = torch.FloatTensor(train_act_probs)
    train_rewards = torch.FloatTensor(train_rewards)
    train_discount_rewards = torch.FloatTensor(discount_rewards)
    return train_obs_v, train_act_v, train_act_probs, train_rewards, train_discount_rewards, reward_mean, reward_std, reward_max, reward_accum, np.min(step_counts), step_counts_mean


def get_ep_reward_logprobs(ep):
    ep_rewards = []
    ep_selected_log_act_probs = []
    for x in ep.steps:
        ep_rewards.append(x.reward)
        ep_selected_log_act_probs.append(np.log(x.action_probs[x.action]))

    return ep_rewards, ep_selected_log_act_probs


def get_epbatch_reward_logprobs(batch):
    ep_rewards_list = []
    ep_selected_log_act_probs_list = []
    for ep in batch:
        ep_rewards, ep_selected_log_act_probs = get_ep_reward_logprobs(ep)
        ep_rewards_list.append(ep_rewards)
        ep_selected_log_act_probs_list.append(ep_selected_log_act_probs)

    return ep_rewards_list, ep_selected_log_act_probs_list


def get_discounted_rewards(rewards, gamma):
    discounted_rewards = []
    for t in range(len(rewards)):
        Gt = 0
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + gamma**pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)

    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards

    return discounted_rewards


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_d(shape):
    mean = torch.zeros(shape)
    cov = torch.eye(shape)
    D = N.MultivariateNormal(mean, cov)
    return D


def sample_d(D, shape, scale=1., grad=True, use_cuda=False, device=torch.device("cuda:0")):
    z = scale * D.sample((shape,))
    if use_cuda:
        z = z.to(device)
    z.requires_grad = grad
    return z


def getShapeDict(model_dict):
    shape_dict = OrderedDict()
    for k, v in model_dict.items():
        shape_dict[k] = v.shape
    return shape_dict


def getWeightTensor(model_dict, CheckExcludeList):
    shape_dict = getShapeDict(model_dict)
    excludeList = [k for k in shape_dict.keys() for name in CheckExcludeList if name in k]
    weight_tensor = torch.zeros(0)
    for k, v in model_dict.items():
        if k not in excludeList:
            weight_tensor = torch.cat([weight_tensor, v.cpu().flatten()], dim=0)
    weight_tensor = weight_tensor.reshape(1, weight_tensor.shape[0])
    return weight_tensor


def getWeightTensorDataset(model_dict_list):
    weight_tensor_list = list(map(getWeightTensor, model_dict_list, itertools.repeat(listToCheck)))
    weight_tensor_concat = torch.cat(weight_tensor_list, dim=0)
    return weight_tensor_concat


def gen_agent_start_pos_list(env):
    goal_pos = (env.width - 2, env.height - 2)
    possible_col = [x for x in range(1, env.width - 1)]
    possible_row = [x for x in range(1, env.height - 1)]
    agent_pos_list = list(product(possible_col, possible_row))
    agent_pos_list = [x for x in agent_pos_list if x != goal_pos]

    agent_dir_list = [x for x in range(4)]
    agent_start_list = list(product(agent_pos_list, agent_dir_list))

    return agent_start_list


def getLoadWeightDict(model_dict, weight_tensor):
    shape_dict = getShapeDict(model_dict)
    excludeList = [k for k in shape_dict.keys() for name in listToCheck if name in k]
    loadWeight_dict = model_dict.copy()
    pool = weight_tensor
    for k, v in model_dict.items():
        if k not in excludeList:
            weight_size = np.prod(list(shape_dict[k]))
            curr = pool[:weight_size]
            loadWeight_dict[k] = curr.resize_(shape_dict[k])
            pool = pool[weight_size:]
    return loadWeight_dict


def concat_layer_bias(l1, b1):
    l1cat = l1.reshape(l1.shape[0],-1)
    cat = torch.cat((l1cat, b1), 1)
    return cat


def convert_GAN2Theta_weights(l1, b1, tensor=False):
    gan_theta = torch.cat((l1.flatten(), b1.flatten()), 0)
    if tensor==False:
        gan_theta = gan_theta.cpu().data.numpy()
    if tensor == True:
        gan_theta = gan_theta.reshape(1, -1)
    return gan_theta


"""A part of the pylabyk library: numpytorch.py at https://github.com/yulkang/pylabyk"""
def block_diag(m):
    """
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    Prepend batch dimensions if needed.
    You can also give a list of matrices.
    :type m: torch.Tensor, list
    :rtype: torch.Tensor
    """
    if type(m) is list:
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

    d = m.dim()
    n = m.shape[-3]
    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]
    m2 = m.unsqueeze(-2)
    eye = attach_dim(torch.eye(n).unsqueeze(-2), d - 3, 1)
    return (m2 * eye).reshape(
        siz0 + torch.Size(torch.tensor(siz1) * n)
    )

def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(
        torch.Size([1] * n_dim_to_prepend)
        + v.shape
        + torch.Size([1] * n_dim_to_append))