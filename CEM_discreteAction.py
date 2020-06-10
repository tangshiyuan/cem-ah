import os, os.path
import pickle
# import gym
# import numpy as np
import argparse
import time

from tensorboardX import SummaryWriter
from datetime import datetime
from collections import OrderedDict

import torch

import gym_minigrid
from gym import wrappers
from gym_minigrid.wrappers import *

import models
import utils


def load_args():
    parser = argparse.ArgumentParser(description='param')
    parser.add_argument('--epochs', default=500, type=int)

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--layer', default=2, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--hidden_s', default=100, type=int) # 20
    # parser.add_argument('--ep_batch', default=1, type=int)
    parser.add_argument('--percentile', default=90, type=int)
    parser.add_argument('--use_cuda', default=True, type=bool)
    parser.add_argument('--device', default='0', type=str)
    parser.add_argument('--comment', default='', type=str)
    parser.add_argument('--write_dir', default='CEM', type=str)

    # parser.add_argument("--port", default=52162)  # for pycharm dev

    args = parser.parse_args()
    return args, parser


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    args, _ = load_args()
    print(args)

    LAYER = vars(args)['layer']
    LEARNING_RATE = vars(args)['lr']
    EPOCHS = vars(args)['epochs']
    HIDDEN_SIZE = vars(args)['hidden_s']
    # BATCH_SIZE = vars(args)['ep_batch']  # number of episodes in a batch
    NET_BATCH = vars(args)['batch_size']  # number of CEM models/episodes (batch size)
    PERCENTILE = vars(args)['percentile']
    use_cuda = args.use_cuda
    DEVICE = args.device
    # monitor_directory = 'monitor_CEM'
    model_directory = 'model_CEM'
    writer_directory = args.write_dir
    exp_name = 'MiniGrid-SimpleCrossingModS9N3-v0'
    COMMENT = args.comment
    SEED = 110

    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    # check number of models in model_directory
    model_count = len([name for name in os.listdir(model_directory) if os.path.isfile(os.path.join(model_directory, name))])
    print('Model Count: {}'.format(model_count))

    config = '_hiddenS' + str(HIDDEN_SIZE) + '_percent' + str(
        PERCENTILE) + '_net' + str(NET_BATCH) + '_layer' + str(
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

    dim_theta = (obs_size + 1) * n_actions

    # Initialize mean and standard deviation
    theta_mean = np.zeros(dim_theta)
    theta_std = np.ones(dim_theta)

    # main policy network
    theta = np.random.multivariate_normal(theta_mean, np.diag(theta_std ** 2))
    if LAYER == 2:
        model = models.NetW_2Layer(args, obs_size, n_actions)  # 2-layer
    if LAYER == 3:
        model = models.NetW_3Layer(args, obs_size, n_actions)  # 3-layer
    model = model.to(device)

    print("Observation Size: {} \t Action Size: {}".format(obs_size, n_actions))

    start_time = time.time()
    # define agent start state
    agent_start_list = [((1, 1), 0)]

    e_batch_buffer = utils.elite_batch_bufferW_msCEM(25)  # 10
    global_eList = []

    total_test_reward_accum = 0
    total_train_reward_accum = 0
    min_reward = round(-env.steps_remaining * 0.1, 1)

    for epoch in range(EPOCHS):
        ratio = (epoch + 1) / (epoch + 2)
        #################################### testing/eval
        test_start = time.time()

        # eval on 10 episodes/10 CEM policies
        eval_ep = 10
        thetas_eval = np.vstack(
            [np.random.multivariate_normal(theta_mean, np.diag(theta_std ** 2)) for _ in range(eval_ep)])
        test_net_batch = []

        for theta in thetas_eval:
            # sample episodes, each state sample BATCH_SIZE episodes, for test, this is 1
            test_batch = utils.iterate_model_batchesWB_msCEM2(env, model, theta, agent_start_list, 1, use_cuda, device, seed=SEED)
            test_net_batch.append(test_batch)

        # filter elite episodes for each state, to just get test batch stats
        test_elite_batch, test_top_batch, test_elite_weights, test_reward_b, test_reward_m, test_reward_std, test_reward_max, test_reward_accum, test_step_counts_min, test_step_counts_m, test_unfit_count = utils.filter_elite_batchW_msCEM_accumR_std(
            test_net_batch, PERCENTILE)

        test_elite_ratio = 1 - test_unfit_count / len(test_net_batch)

        elapsed_time = time.time() - start_time
        test_duration = time.time() - test_start

        print(
            "%d: test_reward_mean=%.5f, test_reward_std=%.5f, test_elite_ratio=%.3f, \tElapsed:%s" % (
                epoch, test_reward_m, test_reward_std,
                test_elite_ratio,
                time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

        #################################### training
        train_start = time.time()
        thetas = np.vstack(
            [np.random.multivariate_normal(theta_mean, np.diag(theta_std ** 2)) for _ in range(NET_BATCH)])
        net_batch = []

        for theta in thetas:
            # sample episodes, each state sample BATCH_SIZE episodes, this is 1 where each weight only evaluated once
            batch = utils.iterate_model_batchesWB_msCEM2(env, model, theta, agent_start_list, 1, use_cuda, device, seed=SEED)
            net_batch.append(batch)

        # filter elite episodes for each state
        elite_batch, top_batch, elite_weights, reward_b, reward_m, reward_std, reward_max, reward_accum, step_counts_min, step_counts_m, unfit_count = utils.filter_elite_batchW_msCEM_accumR_std(
            net_batch, PERCENTILE)

        e_reward_mean, e_step_counts_mean = utils.get_batch_statsW(elite_batch)
        top_reward_mean, top_step_counts_mean = utils.get_batch_statsW(top_batch)

        # get current global elite list stats
        global_eReward, global_eStep = min_reward, 0
        if len(global_eList) > 0:
            global_eReward, global_eStep = utils.get_batch_statsW(global_eList)

        # update global elite list if top batch is better
        if (top_reward_mean > global_eReward):
            global_eList = top_batch
        elif (
                top_reward_mean == global_eReward):  # comment away this to ensure global_reward_m gives equal weightages to all starts
            global_eList.extend(top_batch)

        # if global_eList is too large, take the newest N weights
        max_globalCount = 50
        if len(global_eList) > max_globalCount:
            global_eList = global_eList[-max_globalCount:]

        # add elite_batch to elite buffer only if new elite batch's reward mean >= current overall elite buffer
        # get elite buffer stats
        batch_buffer = e_batch_buffer.get_single_batch()
        eB_reward_mean, eB_step_counts_mean = min_reward, 0
        if len(batch_buffer) > 0:
            eB_reward_mean, eB_step_counts_mean = utils.get_batch_statsW(batch_buffer)
        # adj_ratio = 0.9 * ratio
        adj_ratio = 1
        if (e_reward_mean >= eB_reward_mean * adj_ratio):
            e_batch_buffer.update_buffer(elite_batch)

        elite_ratio = 1 - unfit_count / len(net_batch)

        # construct training data from elite_batch_buffer and global_eList
        elite_thetas = utils.convert_weight_thetas(e_batch_buffer.get_single_batch())
        top_thetas = utils.convert_weight_thetas(global_eList)

        # balance global elite models and elite buffer models count
        g_eB_target = 0.2
        mul_count = max(1, round(
            (g_eB_target * len(elite_thetas)) / (1 - g_eB_target) / len(top_thetas)))
        top_thetas = top_thetas * mul_count
        g_eB_ratio = len(top_thetas) / (len(top_thetas) + len(elite_thetas))
        elite_thetas.extend(top_thetas)

        # Update theta_mean, theta_std
        theta_mean = np.mean(elite_thetas, axis=0)
        theta_std = np.std(elite_thetas, axis=0)

        train_duration = time.time() - train_start
        elapsed_time = time.time() - start_time

        global_eReward, global_eStep = utils.get_batch_statsW(global_eList)

        theta_std_mean = np.mean(theta_std)

        print(
            "%d: reward_mean=%.5f, e_reward_mean=%.5f, global_eReward=%.3f, elite_ratio=%.5f, elite_buffer_count=%i, theta_std_mean=%.3f, \tElapsed:%s" % (
                epoch, reward_m, e_reward_mean, global_eReward, elite_ratio, len(elite_thetas), theta_std_mean,
                time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

        total_test_reward_accum += test_reward_accum
        total_train_reward_accum += reward_accum

        # save the theta_mean and theta_std
        with open(model_path + '.pkl', 'wb') as f:
            pickle.dump((theta_mean, theta_std), f, protocol=pickle.HIGHEST_PROTOCOL)

        writer.add_scalar("train/reward_max", reward_max, epoch)
        writer.add_scalar("train/reward_accum", total_train_reward_accum, epoch)
        writer.add_scalar("train/reward_mean", reward_m, epoch)
        writer.add_scalar("train/reward_std", reward_std, epoch)
        writer.add_scalar("train/step_counts_min", step_counts_min, epoch)
        writer.add_scalar("train/step_counts_mean", step_counts_m, epoch)
        writer.add_scalar("train/elite_ratio", elite_ratio, epoch)
        writer.add_scalar("train/theta_std_mean", theta_std_mean, epoch)

        writer.add_scalar("elite/eliteBuffer_reward_mean", eB_reward_mean, epoch)
        writer.add_scalar("elite/eliteBuffer_step_counts_mean", eB_step_counts_mean, epoch)
        writer.add_scalar("g_elite/global_ep_counts", len(global_eList), epoch)
        writer.add_scalar("g_elite/global_reward", global_eReward, epoch)
        writer.add_scalar("g_elite/global_step_counts", global_eStep, epoch)

        writer.add_scalar("test/reward_max", test_reward_max, epoch)
        writer.add_scalar("test/reward_accum", total_test_reward_accum, epoch)
        writer.add_scalar("test/reward_mean", test_reward_m, epoch)
        writer.add_scalar("test/reward_std", test_reward_std, epoch)
        writer.add_scalar("test/step_counts_min", test_step_counts_min, epoch)
        writer.add_scalar("test/step_counts_mean", test_step_counts_m, epoch)
        writer.add_scalar("test/elite_ratio", test_elite_ratio, epoch)

        writer.add_scalar("duration/train_duration", train_duration, epoch)
        writer.add_scalar("duration/test_duration", test_duration, epoch)

        writer.flush()

    writer.close()
    env.close()