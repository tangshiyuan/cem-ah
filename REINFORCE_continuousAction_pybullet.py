import os, os.path
import pickle
import gym
import numpy as np
import argparse
import time

from tensorboardX import SummaryWriter
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import pybulletgym

import models
import utils

sm = nn.Softmax(dim=1)

def load_args():
    parser = argparse.ArgumentParser(description='param')
    parser.add_argument('--epochs', default=50000, type=int)

    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--layer', default=2, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--hidden_s', default=100, type=int)
    parser.add_argument('--exp_name', default='AntPyBulletEnv-v0', type=str)
    parser.add_argument('--percentile', default=90, type=int)
    parser.add_argument('--use_cuda', default=False, type=bool)
    parser.add_argument('--device', default='0', type=str)
    parser.add_argument('--comment', default='', type=str)
    parser.add_argument('--write_dir', default='REINFORCE', type=str)

    # parser.add_argument("--port", default=52162)  # for pycharm dev

    args = parser.parse_args()
    return args, parser


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args, _ = load_args()
    print(args)

    EPOCHS = vars(args)['epochs']
    HIDDEN_SIZE = vars(args)['hidden_s']
    EXP_NAME = vars(args)['exp_name']
    BATCH_SIZE = vars(args)['batch_size']
    LEARNING_RATE = vars(args)['lr']
    LAYER = vars(args)['layer']
    PERCENTILE = vars(args)['percentile']
    use_cuda = args.use_cuda
    DEVICE = args.device
    # monitor_directory = 'monitor_REINFORCE'
    model_directory = 'model_REINFORCE'
    writer_directory = args.write_dir

    exp_name = EXP_NAME
    COMMENT = args.comment
    SEED = 110

    # hyper parameters
    EPS_START = 0.9  # e-greedy threshold start value
    EPS_END = 0.05  # e-greedy threshold end value
    EPS_DECAY = 200  # e-greedy threshold decay
    GAMMA = 0.9

    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    # check number of models in model_directory
    model_count = len([name for name in os.listdir(model_directory) if os.path.isfile(os.path.join(model_directory, name))])
    print('Model Count: {}'.format(model_count))

    config = '_hiddenS' + str(HIDDEN_SIZE) + '_layer' + str(LAYER) + '_lr' + str(LEARNING_RATE)
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

    MAX_STEPS = env._max_episode_steps

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    print('Active CUDA Device: GPU', torch.cuda.current_device())
    print('Available devices ', torch.cuda.device_count())

    device = torch.device("cuda:" + DEVICE if torch.cuda.is_available() else "cpu")
    if not use_cuda:
        device = torch.device("cpu")
    print(device)

    # main policy network
    if LAYER == 2:
        model = models.NetW_2Layer(args, obs_size, n_actions)  # 2-layer

    if LAYER == 3:
        model = models.NetW_3Layer(args, obs_size, n_actions)  # 3-layer

    model = model.to(device)

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

    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    steps_done = 0

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
            print('======================== Sampling test episodes: state {} ========================'.format((state_num, len(agent_start_list))))
            batch = utils.iterate_model_batches_net_Rstep_bullet(env, model, 10, n_actions, steps_done, EPS_START,
                                                                  EPS_END, EPS_DECAY, use_cuda, device)  # NET_BATCH

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
        all_obs_v = torch.zeros(0, obs_size).to(device)
        all_act_v = torch.zeros(0).to(device)
        # all_act_v = all_act_v.long()
        all_rewards_v = torch.zeros(0).to(device)
        for state_num, agent_start in enumerate(agent_start_list):
            print('======================== Sampling train episodes: state {} ========================'.format((state_num, len(agent_start_list))))
            batch = utils.iterate_model_batches_net_Rstep_bullet(env, model, 1, n_actions, steps_done, EPS_START,
                                                                 EPS_END, EPS_DECAY, use_cuda, device)  # NET_BATCH

            # get batch states and convert states, rewards to tensors
            train_obs_v, train_act_v, train_act_probs, train_rewards, train_discount_rewards, reward_mean_dict[agent_start], reward_std_dict[agent_start], reward_max_dict[agent_start], reward_accum_dict[agent_start], step_counts_min_dict[agent_start], step_counts_mean_dict[agent_start] = utils.get_batch_accumR_std(
                batch, GAMMA)

            obs_v = train_obs_v.reshape(train_obs_v.shape[0], -1)
            # obs_v_norm = (obs_v - obs_v.mean(dim=1, keepdim=True)) / (obs_v.std(dim=1, keepdim=True) + 1e-9)
            # obs_v_norm = obs_v_norm.to(device)

            # train_act_v = train_act_v.to(device)
            train_act_probs = train_act_probs.to(device)
            train_discount_rewards = train_discount_rewards.to(device)

            all_obs_v = torch.cat((all_obs_v, obs_v), 0)
            # all_obs_v = torch.cat((all_obs_v, obs_v_norm), 0)
            # all_act_v = torch.cat((all_act_v, train_act_v), 0)
            all_act_v = torch.cat((all_act_v, train_act_probs), 0)
            all_act_v = all_act_v.long()
            all_rewards_v = torch.cat((all_rewards_v, train_discount_rewards), 0)

            # normalize all_rewards_v to range 0-1
            all_rewards_v_norm = (all_rewards_v - all_rewards_v.min(dim=0, keepdim=True).values) / (
                    all_rewards_v.max(dim=0, keepdim=True).values - all_rewards_v.min(dim=0, keepdim=True).values)

        # define a data batch loader
        reinforce_dataset = torch.utils.data.TensorDataset(all_obs_v, all_act_v, all_rewards_v_norm)
        reinforce_dataloader = torch.utils.data.DataLoader(reinforce_dataset, batch_size=1, shuffle=True)

        ### Train the model for one-epoch
        for i, data in enumerate(reinforce_dataloader, 0):
            optimizer.zero_grad()
            logprob = torch.log(sm(model(data[0])) + 1e-9)
            # selected_logprobs = data[2] * logprob[np.arange(len(data[1])), data[1]]
            selected_logprobs = data[2] * logprob
            policy_loss = -selected_logprobs.mean()

            policy_loss.backward()
            optimizer.step()

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

        logprob = torch.log(sm(model(all_obs_v)) + 1e-9)
        # selected_logprobs = all_rewards_v_norm * logprob[np.arange(len(all_act_v)), all_act_v]
        selected_logprobs = all_rewards_v_norm.reshape(-1, 1) * logprob
        loss = -selected_logprobs.mean()

        elapsed_time = time.time() - start_time
        print("epoch: %d, \tloss: %.4f, \treward_accum: %.4f, \treward_mean: %.4f, \treward_std: %.4f, \tsteps: %d, \tratio: %.4f, \tElapsed:%s" % (
        epoch, loss.item(), reward_accum, reward_m, reward_std, step_counts_m, r_ratio, time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

        # save the model
        torch.save(model.state_dict(), model_path + '_main_net' + '.pth')

        writer.add_scalar("loss/loss", loss.item(), epoch)
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