# model = models.NetW_SingleLayer(args, obs_size, n_actions)
# single start state

import os, os.path
import pickle
# import gym
# import numpy as np
import argparse
import time
import random

from tensorboardX import SummaryWriter
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.nn import functional as F

# import gym_minigrid
# from gym import wrappers
from gym_minigrid.wrappers import *

import models
import utils
from collections import namedtuple

sm = nn.Softmax(dim=1)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

def load_args():
    parser = argparse.ArgumentParser(description='param')
    parser.add_argument('--epochs', default=50000, type=int)

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--layer', default=2, type=int)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--hidden_s', default=100, type=int)
    # parser.add_argument('--ep_batch', default=1, type=int)
    parser.add_argument('--percentile', default=90, type=int)
    parser.add_argument('--use_cuda', default=True, type=bool)
    parser.add_argument('--device', default='0', type=str)
    parser.add_argument('--comment', default='', type=str)
    parser.add_argument('--write_dir', default='DQN', type=str)

    # parser.add_argument("--port", default=52162)  # for pycharm dev

    args = parser.parse_args()
    return args, parser


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args, _ = load_args()
    print(args)

    EPOCHS = vars(args)['epochs']
    HIDDEN_SIZE = vars(args)['hidden_s']
    BATCH_SIZE = vars(args)['batch_size']  # number of training batches sampled from replay buffer
    LEARNING_RATE = vars(args)['lr']
    LAYER = vars(args)['layer']
    PERCENTILE = vars(args)['percentile']
    use_cuda = args.use_cuda
    DEVICE = args.device
    # monitor_directory = 'monitor_DQN'
    model_directory = 'model_DQN'
    writer_directory = args.write_dir
    exp_name = 'MiniGrid-SimpleCrossingModS9N3-v0'
    COMMENT = args.comment
    SEED = 110

    # hyper parameters
    EPS_START = 0.9  # e-greedy threshold start value
    EPS_END = 0.05  # e-greedy threshold end value
    EPS_DECAY = 50000 # e-greedy threshold decay  # 200
    TARGET_UPDATE = 50
    GAMMA = 0.9  # Q-learning discount factor
    TRAIN_BATCH_SIZE = BATCH_SIZE
    TRAIN_STEPS = 1

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

    model_path = os.path.join(model_directory, exp_name+config+event_time+"-"+COMMENT)

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
    if LAYER == 2:
        model = models.NetW_2Layer(args, obs_size, n_actions)  # 2-layer
        target_net = models.NetW_2Layer(args, obs_size, n_actions)  # 2-layer

    if LAYER == 3:
        model = models.NetW_3Layer(args, obs_size, n_actions)  # 3-layer
        target_net = models.NetW_3Layer(args, obs_size, n_actions)  # 3-layer

    model = model.to(device)
    model.apply(init_weights)
    target_net = target_net.to(device)
    target_net.load_state_dict(model.state_dict())
    target_net.eval()

    print("Observation Size: {} \t Action Size: {}".format(obs_size, n_actions))

    mse = nn.MSELoss()

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

    memory = utils.ReplayMemory(10000)
    test_memory = utils.ReplayMemory(1)  # dummy replay memory for test (to reuse code)
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    steps_done = 0
    steps_done_test = 0

    min_reward = round(-env.steps_remaining * 0.1, 1)
    global_reward_max = min_reward
    test_global_reward_max = min_reward

    total_train_reward_accum = 0
    total_test_reward_accum = 0

    eval_ep_batch = 10

    for epoch in range(EPOCHS):
        # print('======================== Sampling episodes ========================')

        # testing/eval ############################
        test_start = time.time()

        # test: sample 10 episodes
        for state_num, agent_start in enumerate(agent_start_list):
            start_pos = agent_start[0]  # col, row
            start_dir = agent_start[1]
            env.set_initial_pos_dir(start_pos, start_dir)
            print('======================== Sampling test episodes: state {} ========================'.format((state_num, len(agent_start_list))))
            test_memory, steps_done_test, batch = utils.iterate_model_batches_net_Rstep_DQN(env, model, test_memory, n_actions, steps_done_test, EPS_START, EPS_END, EPS_DECAY, eval_ep_batch, use_cuda, device, seed=SEED)

            # get batch states and convert states, rewards to tensors
            test_obs_v, test_act_v, test_act_probs, test_rewards, test_discount_rewards, test_reward_mean_dict[agent_start], test_reward_std_dict[agent_start], test_reward_max_dict[agent_start], test_reward_accum_dict[agent_start], test_step_counts_min_dict[agent_start], test_step_counts_mean_dict[agent_start] = utils.get_batch_accumR_std(
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
        for state_num, agent_start in enumerate(agent_start_list):
            start_pos = agent_start[0]  # col, row
            start_dir = agent_start[1]
            env.set_initial_pos_dir(start_pos, start_dir)
            print('======================== Sampling train episodes: state {} ========================'.format((state_num, len(agent_start_list))))
            memory, steps_done, batch = utils.iterate_model_batches_net_Rstep_DQN(env, model, memory, n_actions, steps_done, EPS_START, EPS_END, EPS_DECAY, 1, use_cuda, device, seed=SEED)  # BATCH_SIZE

            # get batch states and convert states, rewards to tensors
            train_obs_v, train_act_v, train_act_probs, train_rewards, train_discount_rewards, reward_mean_dict[agent_start], reward_std_dict[agent_start], reward_max_dict[agent_start], reward_accum_dict[agent_start], step_counts_min_dict[agent_start], step_counts_mean_dict[agent_start] = utils.get_batch_accumR_std(
                batch, GAMMA)

            loss_value = 0
            # Perform one step of the optimization (on the target network) by sampling through memory
            if len(memory) < TRAIN_BATCH_SIZE:
                continue
            for mini_steps in range(TRAIN_STEPS):
                optimizer.zero_grad()
                transitions = memory.sample(TRAIN_BATCH_SIZE)
                # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
                # detailed explanation). This converts batch-array of Transitions
                # to Transition of batch-arrays.
                batch = Transition(*zip(*transitions))

                # Compute a mask of non-final states and concatenate the batch elements
                # (a final state would've been the one after which simulation ended)
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                        batch.next_state)), device=device, dtype=torch.bool)
                non_final_next_states = torch.cat([s for s in batch.next_state
                                                   if s is not None])
                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)

                # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                # columns of actions taken. These are the actions which would've been taken
                # for each batch state according to policy_net
                state_action_values = model(state_batch).gather(1, action_batch)

                # Compute V(s_{t+1}) for all next states.
                # Expected values of actions for non_final_next_states are computed based
                # on the "older" target_net; selecting their best reward with max(1)[0].
                # This is merged based on the mask, such that we'll have either the expected
                # state value or 0 in case the state was final.
                next_state_values = torch.zeros(TRAIN_BATCH_SIZE, device=device)
                next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
                # Compute the expected Q values
                expected_state_action_values = (next_state_values * GAMMA) + reward_batch

                # Compute Huber loss
                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
                loss_value += loss.item()

                # Optimize the model
                loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), 1)
                optimizer.step()

            loss_value = loss_value/TRAIN_STEPS


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

        # Update the target network, copying all weights and biases in DQN
        if epoch % TARGET_UPDATE == 0:
            target_net.load_state_dict(model.state_dict())

        # save the model
        torch.save(model.state_dict(), model_path + '_main_net' + '.pth')
        torch.save(target_net.state_dict(), model_path + '_target_net' + '.pth')

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