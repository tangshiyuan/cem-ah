import os, os.path
import pickle
import gym
import numpy as np
import argparse
import time
import shutil
import gc
import random

from tensorboardX import SummaryWriter
from datetime import datetime
from collections import OrderedDict
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.nn import functional as F
from torch.autograd import Variable

import pybulletgym

import models
import utils


class MemoryBuffer:

    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxSize = size
        self.len = 0

    def sample(self, count):
        """
        samples a random batch from the replay memory buffer
        :param count: batch size
        :return: batch (numpy array)
        """
        batch = []
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)

        s_arr = np.float32([arr[0] for arr in batch])
        a_arr = np.float32([arr[1] for arr in batch])
        r_arr = np.float32([arr[2] for arr in batch])
        s1_arr = np.float32([arr[3] for arr in batch])

        return s_arr, a_arr, r_arr, s1_arr

    def len(self):
        return self.len

    def add(self, s, a, r, s1):
        """
        adds a particular transaction in the memory buffer
        :param s: current state
        :param a: action taken
        :param r: reward received
        :param s1: next state
        :return:
        """
        transition = (s, a, r, s1)
        self.len += 1
        if self.len > self.maxSize:
            self.len = self.maxSize
        self.buffer.append(transition)


class DDPG(nn.Module):

    def __init__(self, args, state_dim, action_dim, action_lim, ram):
        """
        :param state_dim: Dimensions of state (int)
        :param action_dim: Dimension of action (int)
        :param action_lim: Used to limit action in [-action_lim,action_lim]
        :param ram: replay memory buffer object
        :return:
        """
        super(DDPG, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim
        self.ram = ram
        self.iter = 0
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_dim)

        if LAYER == 2:
            self.actor = models.Actor_DDPG_2Layer(args, self.state_dim, self.action_dim, self.action_lim)
            self.target_actor = models.Actor_DDPG_2Layer(args, self.state_dim, self.action_dim, self.action_lim)

            self.critic = models.Critic_DDPG_2Layer(args, self.state_dim, self.action_dim)
            self.target_critic = models.Critic_DDPG_2Layer(args, self.state_dim, self.action_dim)

        if LAYER == 3:
            self.actor = models.Actor_DDPG_3Layer(args, self.state_dim, self.action_dim, self.action_lim)
            self.target_actor = models.Actor_DDPG_3Layer(args, self.state_dim, self.action_dim, self.action_lim)

            self.critic = models.Critic_DDPG_3Layer(args, self.state_dim, self.action_dim)
            self.target_critic = models.Critic_DDPG_3Layer(args, self.state_dim, self.action_dim)

        if LAYER == 4:
            self.actor = models.Actor_DDPG_4Layer(args, self.state_dim, self.action_dim, self.action_lim)
            self.target_actor = models.Actor_DDPG_4Layer(args, self.state_dim, self.action_dim, self.action_lim)

            self.critic = models.Critic_DDPG_4Layer(args, self.state_dim, self.action_dim)
            self.target_critic = models.Critic_DDPG_4Layer(args, self.state_dim, self.action_dim)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), LEARNING_RATE)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), LEARNING_RATE)

        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

    def get_exploitation_action(self, state):
        """
        gets the action from target actor added without exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = Variable(torch.from_numpy(state))
        state = state.to(device)
        action = self.target_actor.forward(state).detach()
        return action.cpu().data.numpy()

    def get_exploration_action(self, state):
        """
        gets the action from actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = Variable(torch.from_numpy(state))
        state = state.to(device)
        action = self.actor.forward(state).detach()
        new_action = action.cpu().data.numpy() + (self.noise.sample() * self.action_lim)
        # new_action = (action.cpu().data.numpy() + np.random.normal(0, NOISE, size=env.action_space.shape[0])).clip(
        #     env.action_space.low, env.action_space.high)

        return new_action

    def optimize(self):
        """
        Samples a random batch from replay memory and performs optimization
        :return:
        """
        s1, a1, r1, s2 = self.ram.sample(TRAIN_BATCH_SIZE)

        s1 = Variable(torch.from_numpy(s1))
        a1 = Variable(torch.from_numpy(a1))
        r1 = Variable(torch.from_numpy(r1))
        s2 = Variable(torch.from_numpy(s2))

        s1 = s1.to(device)
        a1 = a1.to(device)
        r1 = r1.to(device)
        s2 = s2.to(device)

        # ---------------------- optimize critic ----------------------
        # Use target actor exploitation policy here for loss evaluation
        a2 = self.target_actor.forward(s2).detach().to(device)
        next_val = torch.squeeze(self.target_critic.forward(s2, a2).detach())
        y_expected = r1 + GAMMA * next_val
        y_predicted = torch.squeeze(self.critic.forward(s1, a1))
        # compute critic loss, and update the critic
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
        loss_critic_value = loss_critic.item()
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # ---------------------- optimize actor ----------------------
        pred_a1 = self.actor.forward(s1).to(device)
        loss_actor = -1 * torch.sum(self.critic.forward(s1, pred_a1))
        loss_actor_value = loss_actor.item()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        soft_update(self.target_actor, self.actor, TAU)
        soft_update(self.target_critic, self.critic, TAU)

        return loss_critic_value, loss_actor_value

    def save_models(self, episode_count):
        """
        saves the target actor and critic models
        :param episode_count: the count of episodes iterated
        :return:
        """
        torch.save(self.target_actor.state_dict(), model_path + str(episode_count) + '_actor.pt')
        torch.save(self.target_critic.state_dict(), model_path + str(episode_count) + '_critic.pt')
        print('Models saved successfully')

    def load_models(self, episode):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param episode: the count of episodes iterated (used to find the file name)
        :return:
        """
        self.actor.load_state_dict(torch.load(model_path + str(episode) + '_actor.pt'))
        self.critic.load_state_dict(torch.load(model_path + str(episode) + '_critic.pt'))
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        print('Models loaded succesfully')


def load_args():
    parser = argparse.ArgumentParser(description='param')
    parser.add_argument('--epochs', default=50000, type=int)

    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--layer', default=4, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--hidden_s', default=400, type=int)
    parser.add_argument('--exp_name', default='AntPyBulletEnv-v0', type=str)
    parser.add_argument('--percentile', default=90, type=int)
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument('--use_cuda', default=True, type=bool)
    parser.add_argument('--device', default='0', type=str)
    parser.add_argument('--comment', default='', type=str)
    parser.add_argument('--write_dir', default='DDPG', type=str)

    # parser.add_argument("--port", default=52162)  # for pycharm dev

    args = parser.parse_args()
    return args, parser


def soft_update(target, source, tau):
    """
    Copies the parameters from source network (x) to target network (y) using the below update
    y = TAU*x + (1 - TAU)*y
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    """
    Copies the parameters from source network to target network
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def save_training_checkpoint(state, is_best, episode_count):
    """
    Saves the models, with all training parameters intact
    :param state:
    :param is_best:
    :param filename:
    :return:
    """
    filename = str(episode_count) + 'checkpoint.path.rar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:

    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args, _ = load_args()
    print(args)

    EPOCHS = vars(args)['epochs']
    HIDDEN_SIZE = vars(args)['hidden_s']
    EXP_NAME = vars(args)['exp_name']
    LEARNING_RATE = vars(args)['lr']
    LAYER = vars(args)['layer']
    BATCH_SIZE = vars(args)['batch_size']  # number of training batches sampled from replay buffer
    PERCENTILE = vars(args)['percentile']
    use_cuda = args.use_cuda
    DEVICE = args.device
    # monitor_directory = 'monitor_DDPG'
    model_directory = 'model_DDPG'
    writer_directory = args.write_dir

    exp_name = EXP_NAME
    COMMENT = args.comment
    SEED = 110

    TRAIN_BATCH_SIZE = BATCH_SIZE
    GAMMA = 0.99  # 0.9
    TAU = 0.005  # 1e-2
    NOISE = vars(args)['expl_noise']

    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    # check number of models in model_directory
    model_count = len(
        [name for name in os.listdir(model_directory) if os.path.isfile(os.path.join(model_directory, name))])
    print('Model Count: {}'.format(model_count))

    config = '_hiddenS' + str(HIDDEN_SIZE) + '_batch' + str(BATCH_SIZE) + '_layer' + str(
        LAYER) + '_lr' + str(LEARNING_RATE)
    event_time = '_' + datetime.now().strftime("%Y%m%d_%H%M%S")

    # path = os.path.join(monitor_directory, exp_name + config + event_time)
    # if not os.path.exists(path):
    #     os.makedirs(path)

    writer_path = os.path.join(writer_directory, exp_name + config + event_time + "-" + COMMENT)
    if not os.path.exists(writer_path):
        os.makedirs(writer_path)

    model_path = os.path.join(model_directory, exp_name + config + event_time + "-" + COMMENT)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    writer = SummaryWriter(writer_path)

    env = gym.make(exp_name)

    MAX_STEPS = env._max_episode_steps
    MAX_BUFFER = 1000000

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    print('Active CUDA Device: GPU', torch.cuda.current_device())
    print('Available devices ', torch.cuda.device_count())

    device = torch.device("cuda:" + DEVICE if torch.cuda.is_available() else "cpu")
    if not use_cuda:
        device = torch.device("cpu")
    print(device)

    S_DIM = obs_size
    A_DIM = n_actions
    A_MAX = env.action_space.high

    print(' State Dimensions :- ', S_DIM)
    print(' Action Dimensions :- ', A_DIM)
    print(' Action Max :- ', A_MAX)

    ram = MemoryBuffer(MAX_BUFFER)
    trainer = DDPG(args, S_DIM, A_DIM, A_MAX, ram)
    trainer = trainer.to(device)

    print("Observation Size: {} \t Action Size: {}".format(obs_size, n_actions))

    start_time = time.time()
    # define agent start state
    agent_start_list = [(0,0)]

    reward_mean_dict = OrderedDict()
    reward_std_dict = OrderedDict()
    step_counts_mean_dict = OrderedDict()
    reward_max_dict = OrderedDict()
    reward_accum_dict = OrderedDict()
    step_counts_min_dict = OrderedDict()

    test_reward_mean_dict = OrderedDict()
    test_reward_std_dict = OrderedDict()
    test_step_counts_mean_dict = OrderedDict()
    test_reward_max_dict = OrderedDict()
    test_reward_accum_dict = OrderedDict()
    test_step_counts_min_dict = OrderedDict()

    for agent_start in agent_start_list:
        reward_mean_dict[agent_start] = 0
        reward_std_dict[agent_start] = 0
        step_counts_mean_dict[agent_start] = 0
        reward_max_dict[agent_start] = 0
        reward_accum_dict[agent_start] = 0
        step_counts_min_dict[agent_start] = 0

        test_reward_mean_dict[agent_start] = 0
        test_reward_std_dict[agent_start] = 0
        test_step_counts_mean_dict[agent_start] = 0
        test_reward_max_dict[agent_start] = 0
        test_reward_accum_dict[agent_start] = 0
        test_step_counts_min_dict[agent_start] = 0

    min_reward = -1.00e5
    global_reward_max = min_reward
    test_global_reward_max = min_reward

    total_train_reward_accum = 0
    total_test_reward_accum = 0

    for epoch in range(EPOCHS):
        # testing/eval ############################
        test_start = time.time()

        # test: sample 10 episodes
        for state_num, agent_start in enumerate(agent_start_list):
            print('======================== Sampling test episodes: state {} ========================'.format(
                (state_num, len(agent_start_list))))
            batch = utils.iterate_model_batches_net_Rstep_bullet_DDPG(env, trainer, ram, 10, use_cuda, device, train=False)

            # get batch states and convert states, rewards to tensors
            test_obs_v, test_act_v, test_act_probs, test_rewards, test_discount_rewards, test_reward_mean_dict[
                agent_start], test_reward_std_dict[agent_start], test_reward_max_dict[agent_start], \
            test_reward_accum_dict[agent_start], test_step_counts_min_dict[agent_start], test_step_counts_mean_dict[
                agent_start] = utils.get_batch_accumR_std(
                batch, GAMMA)

        test_duration = time.time() - test_start

        test_reward_max = np.mean(list(test_reward_max_dict.values()))
        test_reward_accum = np.sum(list(test_reward_accum_dict.values()))
        test_reward_m = np.mean(list(test_reward_mean_dict.values()))
        test_reward_std = np.max(list(test_reward_std_dict.values()))
        test_step_counts_min = np.mean(list(test_step_counts_min_dict.values()))
        test_step_counts_m = np.mean(list(test_step_counts_mean_dict.values()))

        if test_reward_m > test_global_reward_max:
            test_global_reward_max = test_reward_m

        test_r_ratio = test_reward_m / test_global_reward_max
        total_test_reward_accum += test_reward_accum

        elapsed_time = time.time() - start_time
        print(
            "epoch: %d, \ttest_reward_accum: %.4f, \ttest_reward_m: %.4f, \ttest_reward_std: %.4f, \ttest_step_counts_m: %d, \ttest_r_ratio: %.4f, \tElapsed:%s" % (
                epoch, test_reward_accum, test_reward_m, test_reward_std, test_step_counts_m, test_r_ratio,
                time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

        # training ############################
        train_start = time.time()

        # sample a batch of episodes
        all_obs_v = torch.zeros(0, obs_size).to(device)
        all_act_v = torch.zeros(0).to(device)
        all_rewards_v = torch.zeros(0).to(device)
        for state_num, agent_start in enumerate(agent_start_list):
            print('======================== Sampling train episodes: state {} ========================'.format(
                (state_num, len(agent_start_list))))
            batch = utils.iterate_model_batches_net_Rstep_bullet_DDPG(env, trainer, ram, 1, use_cuda, device,
                                                               train=True)

            # get batch states and convert states, rewards to tensors
            train_obs_v, train_act_v, train_act_probs, train_rewards, train_discount_rewards, reward_mean_dict[
                agent_start], reward_std_dict[agent_start], reward_max_dict[agent_start], reward_accum_dict[
                agent_start], step_counts_min_dict[agent_start], step_counts_mean_dict[
                agent_start] = utils.get_batch_accumR_std(
                batch, GAMMA)

            # perform optimization (train once every episode, not every step)
            loss_critic_value, loss_actor_value = trainer.optimize()

        train_duration = time.time() - train_start

        reward_max = np.mean(list(reward_max_dict.values()))
        reward_accum = np.sum(list(reward_accum_dict.values()))
        reward_m = np.mean(list(reward_mean_dict.values()))
        reward_std = np.max(list(reward_std_dict.values()))
        step_counts_min = np.mean(list(step_counts_min_dict.values()))
        step_counts_m = np.mean(list(step_counts_mean_dict.values()))

        if reward_m > global_reward_max:
            global_reward_max = reward_m

        r_ratio = reward_m / global_reward_max
        total_train_reward_accum += reward_accum

        elapsed_time = time.time() - start_time
        print(
            "epoch: %d, \tloss_critic: %.4f, \tloss_actor: %.4f, \treward_accum: %.4f, \treward_mean: %.4f, \treward_std: %.4f, \tsteps: %d, \tratio: %.4f, \tElapsed:%s" % (
                epoch, loss_critic_value, loss_actor_value, reward_accum, reward_m, reward_std, step_counts_m, r_ratio,
                time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

        # save the model
        trainer.save_models(episode_count=0)  # to overwrite. set episode_count=epoch to disable overwrite

        writer.add_scalar("loss/loss_critic", loss_critic_value, epoch)
        writer.add_scalar("loss/loss_actor", loss_actor_value, epoch)
        writer.add_scalar("train/reward_max", reward_max, epoch)
        writer.add_scalar("train/reward_mean", reward_m, epoch)
        writer.add_scalar("train/reward_std", reward_std, epoch)
        writer.add_scalar("train/reward_accum", total_train_reward_accum, epoch)
        writer.add_scalar("train/r_ratio", r_ratio, epoch)
        writer.add_scalar("train/global_reward_max", global_reward_max, epoch)
        writer.add_scalar("train/step_counts_min", step_counts_min, epoch)
        writer.add_scalar("train/step_counts_mean", step_counts_m, epoch)

        writer.add_scalar("test/reward_max", test_reward_max, epoch)
        writer.add_scalar("test/reward_mean", test_reward_m, epoch)
        writer.add_scalar("test/reward_std", test_reward_std, epoch)
        writer.add_scalar("test/reward_accum", total_test_reward_accum, epoch)
        writer.add_scalar("test/r_ratio", test_r_ratio, epoch)
        writer.add_scalar("test/global_reward_max", test_global_reward_max, epoch)
        writer.add_scalar("test/step_counts_min", test_step_counts_min, epoch)
        writer.add_scalar("test/step_counts_mean", test_step_counts_m, epoch)

        writer.add_scalar("duration/train_duration", train_duration, epoch)
        writer.add_scalar("duration/test_duration", test_duration, epoch)

        writer.flush()

    writer.close()
    env.close()
