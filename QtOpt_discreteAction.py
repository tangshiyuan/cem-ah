import os, os.path
import pickle
# import gym
import numpy as np
import argparse
import time
import random
import math


from tensorboardX import SummaryWriter
from datetime import datetime
from collections import OrderedDict
from scipy.special import softmax

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions import Categorical

from gym_minigrid.wrappers import *

import models
import utils
from collections import namedtuple

sm = nn.Softmax(dim=1)
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class QT_Opt(nn.Module):
    def __init__(self, replay_buffer, hidden_dim, action_lim, q_lr=3e-4, cem_update_itr=2, select_num=6, num_samples=64):
        super(QT_Opt, self).__init__()
        self.num_samples = num_samples
        self.select_num = select_num
        self.cem_update_itr = cem_update_itr
        self.replay_buffer = replay_buffer
        self.noise = OrnsteinUhlenbeckActionNoise(n_actions)
        self.action_lim = action_lim

        if LAYER == 2:
            self.qnet = models.Critic_DDPG_2Layer(args, obs_size, n_actions).to(device)
            self.target_qnet1 = models.Critic_DDPG_2Layer(args, obs_size, n_actions).to(device)
            self.target_qnet2 = models.Critic_DDPG_2Layer(args, obs_size, n_actions).to(device)

        if LAYER == 3:
            self.qnet = models.Critic_DDPG_3Layer(args, obs_size, n_actions).to(device)
            self.target_qnet1 = models.Critic_DDPG_3Layer(args, obs_size, n_actions).to(device)
            self.target_qnet2 = models.Critic_DDPG_3Layer(args, obs_size, n_actions).to(device)

        self.cem = models.CEM_QtOpt(n_actions, action_lim)

        self.target_qnet1.load_state_dict(self.qnet.state_dict())
        self.target_qnet2.load_state_dict(self.qnet.state_dict())
        self.target_qnet1.eval()
        self.target_qnet2.eval()

        self.q_optimizer = optim.Adam(self.qnet.parameters(), lr=q_lr)
        self.step_cnt = 0

    def update(self, batch_size, gamma=0.9, soft_tau=1e-2, update_delay=50):
        transitions = self.replay_buffer.sample(batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done)

        self.step_cnt += 1

        predict_q = self.qnet(state_batch, action_batch)  # predicted Q(s,a) value

        # get argmax_a' from the CEM for the target Q(s', a')
        new_next_action = []

        for i in range(batch_size):  # batch of states, use them one by one, to prevent the lack of memory
            new_next_action.append(self.cem_optimal_action_exploit(next_state_batch[i].reshape(1, -1)))
        new_next_action = torch.FloatTensor(new_next_action).to(device)

        target_q_min = torch.min(self.target_qnet1(next_state_batch, new_next_action),
                                 self.target_qnet2(next_state_batch, new_next_action))
        target_q = reward_batch + (1 - done_batch) * gamma * target_q_min

        q_loss = ((predict_q - target_q.detach()) ** 2).mean()  # MSE loss, note that original paper uses cross-entropy loss
        # print(q_loss)
        loss_value = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # update the target nets, according to original paper:
        # one with Polyak averaging, another with lagged/delayed update
        self.target_qnet1 = self.target_soft_update(self.qnet, self.target_qnet1, soft_tau)
        self.target_qnet2 = self.target_delayed_update(self.qnet, self.target_qnet2, update_delay)
        return loss_value

    def cem_optimal_action_exploit(self, state):
        '''
        evaluate action wrt Q(s,a) to select the optimal using CEM
        gets the action from target actor added without exploration noise
        '''
        self.cem.initialize()  # the critical line
        cuda_states = torch.cat([state] * qt_opt.num_samples)
        for itr in range(self.cem_update_itr):
            actions = self.cem.sample_multi(self.num_samples)
            actions = actions * self.action_lim
            action_probs = softmax(actions, axis=1)
            q_values = self.target_qnet1(cuda_states,
                                         torch.FloatTensor(action_probs).to(device)).detach().cpu().numpy().reshape(
                -1)  # 2 dim to 1 dim
            max_idx = q_values.argsort()[-1]  # select one maximal q
            idx = q_values.argsort()[-int(self.select_num):]  # select top maximum q
            selected_action_probs = action_probs[idx]
            _, _ = self.cem.update(selected_action_probs)
        optimal_action = action_probs[max_idx]
        return optimal_action

    def cem_optimal_action_explore(self, state):
        '''
        evaluate action wrt Q(s,a) to select the optimal using CEM
        gets the action from actor added with exploration noise
        '''
        self.cem.initialize()  # the critical line
        cuda_states = torch.cat([state] * qt_opt.num_samples)
        for itr in range(self.cem_update_itr):
            actions = self.cem.sample_multi(self.num_samples)
            actions = actions * self.action_lim
            action_probs = softmax(actions, axis=1)
            q_values = self.qnet(cuda_states,
                                 torch.FloatTensor(action_probs).to(device)).detach().cpu().numpy().reshape(
                -1)  # 2 dim to 1 dim
            max_idx = q_values.argsort()[-1]  # select one maximal q
            idx = q_values.argsort()[-int(self.select_num):]  # select top maximum q
            selected_action_probs = action_probs[idx]
            _, _ = self.cem.update(selected_action_probs)
        optimal_action = action_probs[max_idx]
        return optimal_action

    def target_soft_update(self, net, target_net, soft_tau):
        ''' Soft update the target net '''
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return target_net

    def target_delayed_update(self, net, target_net, update_delay):
        ''' delayed update the target net '''
        if self.step_cnt % update_delay == 0:
            for target_param, param in zip(target_net.parameters(), net.parameters()):
                target_param.data.copy_(  # copy data value into target parameters
                    param.data
                )
        return target_net

    def save_model(self, path):
        torch.save(self.qnet.state_dict(), path)
        torch.save(self.target_qnet1.state_dict(), path)
        torch.save(self.target_qnet2.state_dict(), path)

    def load_model(self, path):
        self.qnet.load_state_dict(torch.load(path))
        self.target_qnet1.load_state_dict(torch.load(path))
        self.target_qnet2.load_state_dict(torch.load(path))
        self.qnet.eval()
        self.target_qnet1.eval()
        self.target_qnet2.eval()


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


def load_args():
    parser = argparse.ArgumentParser(description='param')
    parser.add_argument('--epochs', default=50000, type=int)

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--layer', default=2, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--hidden_s', default=100, type=int)
    parser.add_argument('--percentile', default=90, type=int)
    parser.add_argument('--net_batch', default=25, type=int)
    parser.add_argument('--use_cuda', default=True, type=bool)
    parser.add_argument('--device', default='0', type=str)
    parser.add_argument('--comment', default='', type=str)
    parser.add_argument('--write_dir', default='QtOpt', type=str)

    # parser.add_argument("--port", default=52162)  # for pycharm dev

    args = parser.parse_args()
    return args, parser


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args, _ = load_args()
    print(args)

    LEARNING_RATE = vars(args)['lr']
    LAYER = vars(args)['layer']
    EPOCHS = vars(args)['epochs']
    HIDDEN_SIZE = vars(args)['hidden_s']
    BATCH_SIZE = vars(args)['batch_size']  # batch size to sample from replay buffer
    NET_BATCH = vars(args)['net_batch']  # number of CEM action samples
    PERCENTILE = vars(args)['percentile']
    use_cuda = args.use_cuda
    DEVICE = args.device
    # monitor_directory = 'monitor_QtOpt'
    model_directory = 'model_QtOpt'
    writer_directory = args.write_dir
    exp_name = 'MiniGrid-SimpleCrossingModS9N3-v0'
    COMMENT = args.comment
    SEED = 110
    A_MAX = 1
    replay_buffer_size = 5e5
    replay_buffer = ReplayBuffer(replay_buffer_size)

    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    # check number of models in model_directory
    model_count = len([name for name in os.listdir(model_directory) if os.path.isfile(os.path.join(model_directory, name))])
    print('Model Count: {}'.format(model_count))

    config = '_hiddenS' + str(HIDDEN_SIZE) + '_batch' + str(BATCH_SIZE) + '_layer' + str(
        LAYER) + '_lr' + str(LEARNING_RATE)
    event_time = '_' + datetime.now().strftime("%Y%m%d_%H%M%S")

    # path = os.path.join(monitor_directory, exp_name + config + event_time)
    # if not os.path.exists(path):
    #     os.makedirs(path)

    writer_path = os.path.join(writer_directory, exp_name+config+event_time+"-"+COMMENT)
    if not os.path.exists(writer_path):
        os.makedirs(writer_path)

    model_path = os.path.join(model_directory, exp_name + config + event_time + "-" + COMMENT)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    writer = SummaryWriter(writer_path)

    env = gym.make(exp_name)
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)  # Get rid of the 'mission' field
    env.set_max_steps(10000)
    env.seed(SEED)  # set seed to 110
    obs = env.reset()  # This now produces an RGB tensor only

    obs_size = obs.size
    n_actions = env.action_space.n

    print('Active CUDA Device: GPU', torch.cuda.current_device())
    print('Available devices ', torch.cuda.device_count())

    device = torch.device("cuda:" + DEVICE if torch.cuda.is_available() else "cpu")
    if not use_cuda:
        device = torch.device("cpu")
    print(device)

    # main policy network
    qt_opt = QT_Opt(replay_buffer, HIDDEN_SIZE, A_MAX, q_lr=LEARNING_RATE, select_num=int(NET_BATCH * (1 - (PERCENTILE / 100))),
                    num_samples=NET_BATCH)
    qt_opt = qt_opt.to(device)

    print("Observation Size: {} \t Action Size: {}".format(obs_size, n_actions))

    start_time = time.time()
    # define agent start state
    agent_start_list = [((1, 1), 0)]

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

    min_reward = round(-env.steps_remaining * 0.1, 1)
    global_reward_max = min_reward
    test_global_reward_max = min_reward

    total_train_reward_accum = 0
    total_test_reward_accum = 0

    eval_ep_batch = 10

    for epoch in range(EPOCHS):
        # testing/eval ############################
        test_start = time.time()

        # test: sample 10 episodes
        for state_num, agent_start in enumerate(agent_start_list):
            start_pos = agent_start[0]  # col, row
            start_dir = agent_start[1]
            env.set_initial_pos_dir(start_pos, start_dir)
            batch_episode_reward = []
            batch_steps = []

            print('======================== Sampling test episodes: state {} ========================'.format(
                (state_num, len(agent_start_list))))
            for episodes in range(eval_ep_batch):
                # Initialize the environment and state
                episode_reward = 0.0  # added
                steps = 0

                env.seed(SEED)
                obs = env.reset()
                obs_v = torch.FloatTensor([obs.flatten()])
                obs_v = obs_v.to(device)
                # standardize observation vector
                obs_v_norm = (obs_v - obs_v.mean(dim=1, keepdim=True)) / obs_v.std(dim=1, keepdim=True)
                while True:
                    # Select and perform an action
                    act_probs = qt_opt.cem_optimal_action_exploit(obs_v_norm)
                    action = np.random.choice(len(act_probs), p=act_probs)

                    next_obs, reward, is_done, _ = env.step(action)
                    episode_reward += reward
                    action_v = torch.tensor([act_probs], device=device).reshape(1, -1)
                    action_v = action_v.float()
                    reward_v = torch.tensor([reward], device=device).reshape(1, -1)
                    done = torch.FloatTensor([np.float32(is_done)]).to(device).reshape(1, -1)

                    next_obs_v = torch.FloatTensor([next_obs.flatten()])
                    next_obs_v = next_obs_v.to(device)
                    # standardize observation vector
                    next_obs_v_norm = (next_obs_v - next_obs_v.mean(dim=1, keepdim=True)) / next_obs_v.std(dim=1,
                                                                                                           keepdim=True)

                    # Move to the next state
                    obs_v_norm = next_obs_v_norm
                    steps += 1
                    if is_done:
                        break

                batch_episode_reward.append(episode_reward)
                batch_steps.append(steps)

            test_reward_mean_dict[agent_start] = np.mean(batch_episode_reward)
            test_reward_std_dict[agent_start] = np.std(batch_episode_reward)
            test_reward_max_dict[agent_start] = np.max(batch_episode_reward)
            test_reward_accum_dict[agent_start] = np.sum(batch_episode_reward)
            test_step_counts_min_dict[agent_start] = np.min(batch_steps)
            test_step_counts_mean_dict[agent_start] = np.mean(batch_steps)

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

        loss_value = 0.0
        # sample one episode and update
        for state_num, agent_start in enumerate(agent_start_list):
            start_pos = agent_start[0]  # col, row
            start_dir = agent_start[1]
            env.set_initial_pos_dir(start_pos, start_dir)
            batch_episode_reward = []
            batch_steps = []
            print('======================== Sampling train episodes: state {} ========================'.format(
                (state_num, len(agent_start_list))))
            for episodes in range(1):
                # Initialize the environment and state
                episode_reward = 0.0  # added
                steps = 0

                env.seed(SEED)
                obs = env.reset()
                obs_v = torch.FloatTensor([obs.flatten()])
                obs_v = obs_v.to(device)
                # standardize observation vector
                obs_v_norm = (obs_v - obs_v.mean(dim=1, keepdim=True)) / obs_v.std(dim=1, keepdim=True)
                while True:
                    # Select and perform an action
                    act_probs = qt_opt.cem_optimal_action_explore(obs_v_norm)
                    action = np.random.choice(len(act_probs), p=act_probs)

                    next_obs, reward, is_done, _ = env.step(action)
                    episode_reward += reward
                    action_v = torch.tensor([act_probs], device=device).reshape(1, -1)
                    action_v = action_v.float()
                    reward_v = torch.tensor([reward], device=device).reshape(1, -1)
                    done = torch.FloatTensor([np.float32(is_done)]).to(device).reshape(1, -1)

                    next_obs_v = torch.FloatTensor([next_obs.flatten()])
                    next_obs_v = next_obs_v.to(device)
                    # standardize observation vector
                    next_obs_v_norm = (next_obs_v - next_obs_v.mean(dim=1, keepdim=True)) / next_obs_v.std(dim=1,
                                                                                                           keepdim=True)

                    # # Store the transition in replay buffer
                    replay_buffer.push(obs_v_norm, action_v, reward_v, next_obs_v_norm, done)

                    # Move to the next state
                    obs_v_norm = next_obs_v_norm
                    steps += 1
                    if is_done:
                        break

                # Perform one step of the optimization (on the target network)
                if len(replay_buffer) >= BATCH_SIZE:
                    loss_value = qt_opt.update(BATCH_SIZE)

                batch_episode_reward.append(episode_reward)
                batch_steps.append(steps)

            reward_mean_dict[agent_start] = np.mean(batch_episode_reward)
            reward_std_dict[agent_start] = np.std(batch_episode_reward)
            reward_max_dict[agent_start] = np.max(batch_episode_reward)
            reward_accum_dict[agent_start] = np.sum(batch_episode_reward)
            step_counts_min_dict[agent_start] = np.min(batch_steps)
            step_counts_mean_dict[agent_start] = np.mean(batch_steps)


        train_duration = time.time() - train_start

        reward_max = np.mean(list(reward_max_dict.values()))
        reward_accum = np.sum(list(reward_accum_dict.values()))
        reward_m = np.mean(list(reward_mean_dict.values()))
        reward_std = np.max(list(reward_std_dict.values()))
        step_counts_min = np.mean(list(step_counts_min_dict.values()))
        step_counts_m = np.mean(list(step_counts_mean_dict.values()))

        if reward_m > global_reward_max:
            global_reward_max = reward_m

        r_ratio = reward_m/global_reward_max
        total_train_reward_accum += reward_accum

        elapsed_time = time.time() - start_time
        print("epoch: %d, \tloss: %.4f, \treward_accum: %.4f, \treward_mean: %.4f, \treward_std: %.4f, \tsteps: %d, \tratio: %.4f, \tElapsed:%s" % (
        epoch, loss_value, reward_accum, reward_m, reward_std, step_counts_m, r_ratio, time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

        # save the model
        torch.save(qt_opt.qnet.state_dict(), model_path + '_qnet' + '.pth')
        torch.save(qt_opt.target_qnet1.state_dict(), model_path + '_target_qnet1' + '.pth')
        torch.save(qt_opt.target_qnet2.state_dict(), model_path + '_target_qnet2' + '.pth')

        writer.add_scalar("loss/loss", loss_value, epoch)
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

        # remove the retained graph
        loss = None

    writer.close()
    env.close()