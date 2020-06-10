import os
import gym
import numpy as np
import argparse
import time
import pickle

from tensorboardX import SummaryWriter
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from scipy.special import softmax

import gym_minigrid
from gym import wrappers
from gym_minigrid.wrappers import *

import models
import utils

sm = nn.Softmax(dim=1)

def load_args():
    parser = argparse.ArgumentParser(description='param')
    # parser.add_argument('--exp', default='maze', type=str, help='experiment')
    parser.add_argument('--z', default=200, type=int, help='latent space width')
    parser.add_argument('--ze', default=100, type=int, help='encoder dimension')
    parser.add_argument('--epochs', default=500, type=int)

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--layer', default=2, type=int)
    parser.add_argument('--hidden_s', default=100, type=int)
    # parser.add_argument('--ep_batch', default=1, type=int)
    parser.add_argument('--percentile', default=90, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--update_ratio', default=100, type=int)
    parser.add_argument('--update_decay', default=0.05, type=float)
    parser.add_argument('--use_cuda', default=True, type=bool)
    parser.add_argument('--device', default='0', type=str)
    parser.add_argument('--comment', default='', type=str)
    parser.add_argument('--write_dir', default='CEM-AH', type=str)
    # parser.add_argument("--port", default=52162)  # for pycharm dev

    args = parser.parse_args()
    return args, parser


def sample_fake_eObs_netBatch_mini(model, obs_v, net_batch, obs_per_net, obs_size, n_actions, use_cuda, device, show=True):
    if show:
        print('======================== Generating fake action probabilities ========================')
    start_time = time.time()

    mini_obs_v = obs_v  # obs_v here is already standardized
    # empty tensors initialise
    fake_obs_v = torch.zeros(0, obs_size)
    fake_act_probs_v = torch.zeros(0, n_actions)

    if use_cuda:
        fake_obs_v = fake_obs_v.to(device)
        fake_act_probs_v = fake_act_probs_v.to(device)

    for i in range(net_batch):
        model.sample_noise()  # sample noise z and generate random network

        net_obs_v = mini_obs_v[0:obs_per_net]
        mini_obs_v = mini_obs_v[obs_per_net:]

        fake_obs_v = torch.cat((fake_obs_v, net_obs_v), 0)

        act_pobs_v = model(net_obs_v)
        fake_act_probs_v = torch.cat((fake_act_probs_v, act_pobs_v), 0)

    elapsed_time = time.time() - start_time
    if show:
        print("Fake tuples=%d, \tDuration:%s "
              % (fake_obs_v.shape[0], time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

    return fake_obs_v, fake_act_probs_v


# create real dataloader for Pre-training
def create_real_batch_loader(obs_v, act_probs_v, batch_size, use_cuda, device, shuffle=False):
    # concat observation space with action probabilities
    obs_act_cat = torch.cat((obs_v, act_probs_v), 1)

    dis_inputs = obs_act_cat
    dis_targets = Variable(torch.ones(obs_act_cat.shape[0], 1))*0.9

    if use_cuda:
        dis_targets = dis_targets.to(device)

    # define a data batch loader
    dis_dataset = torch.utils.data.TensorDataset(obs_v, act_probs_v, dis_targets)
    dis_dataloader = torch.utils.data.DataLoader(dis_dataset, batch_size=batch_size, shuffle=shuffle)

    return obs_v, act_probs_v, dis_targets, dis_dataloader


# create batch loader for Discriminator
def create_dis_batch_loader(obs_v, act_probs_v, fake_obs_v, fake_act_pobs_v, use_cuda, device):
    # concat observation space with action probabilities
    obs_act_cat = torch.cat((obs_v, act_probs_v), 1)
    fake_obs_act_cat = torch.cat((fake_obs_v, fake_act_pobs_v), 1)

    # merge the real and fake data
    dis_inputs = torch.cat((obs_act_cat, fake_obs_act_cat), 0)
    real_targets = Variable(torch.ones(obs_act_cat.shape[0], 1))*0.9
    fake_targets = Variable(torch.ones(fake_obs_act_cat.shape[0], 1))*0.1
    dis_targets = torch.cat((real_targets, fake_targets), 0)

    if use_cuda:
        dis_targets = dis_targets.to(device)

    ratio = min(obs_act_cat.shape[0],fake_obs_act_cat.shape[0])/(obs_act_cat.shape[0]+fake_obs_act_cat.shape[0])
    print("Real Tuples: {}, Fake Tuples: {}, Ratio: {}".format(obs_act_cat.shape[0], fake_obs_act_cat.shape[0], ratio))

    # define a GAN data batch loader
    dis_dataset = torch.utils.data.TensorDataset(dis_inputs, dis_targets)
    dis_dataloader = torch.utils.data.DataLoader(dis_dataset, batch_size=gan_batch, shuffle=True)

    return dis_inputs, dis_targets, dis_dataloader


def sample_train_msCEM(e_batch_buffer_msCEM, global_eList_msCEM, env, msCEM_model, net_batch, ep_batch, theta_mean,
                       theta_std, use_cuda, device, seed):
    min_reward = round(-env.steps_remaining * 0.1, 1)

    thetas = np.vstack([np.random.multivariate_normal(theta_mean, np.diag(theta_std ** 2)) for _ in range(net_batch)])
    msCEM_net_batch = []
    for theta in thetas:
        # sample episodes, each state sample BATCH_SIZE episodes  # 1
        batch = utils.iterate_model_batchesWB_msCEM2(env, msCEM_model, theta, agent_start_list, ep_batch, use_cuda,
                                                     device, seed=seed)
        msCEM_net_batch.append(batch)
    # filter elite episodes for each state
    elite_batch, top_batch, elite_weights, reward_b, reward_m, ave_reward_std, reward_max, reward_accum, step_counts_min, step_counts_m, unfit_count = utils.filter_elite_batchW_msCEM_accumR_std(
        msCEM_net_batch, PERCENTILE)

    e_reward_mean, e_step_counts_mean = utils.get_batch_statsW(elite_batch)
    top_reward_mean, top_step_counts_mean = utils.get_batch_statsW(top_batch)

    # get current global elite list stats
    global_eReward, global_eStep = min_reward, 0
    if len(global_eList_msCEM) > 0:
        global_eReward, global_eStep = utils.get_batch_statsW(global_eList_msCEM)

    # update global elite list if top batch is better
    if (top_reward_mean > global_eReward):
        global_eList_msCEM = top_batch
    elif (
            top_reward_mean == global_eReward):
        global_eList_msCEM.extend(top_batch)

    # if global_eList is too large, take the newest N weights
    max_globalCount = 50
    if len(global_eList_msCEM) > max_globalCount:
        global_eList_msCEM = global_eList_msCEM[-max_globalCount:]

    if len(global_eList_msCEM) > 0:
        global_eReward, global_eStep = utils.get_batch_statsW(global_eList_msCEM)

    # get elite buffer stats
    batch_buffer = e_batch_buffer_msCEM.get_single_batch()
    eB_reward_mean, eB_step_counts_mean = min_reward, 0
    if len(batch_buffer) > 0:
        eB_reward_mean, eB_step_counts_mean = utils.get_batch_statsW(batch_buffer)
    # adj_ratio = 0.9 * ratio
    adj_ratio = 1
    # add elite_batch to elite buffer only if new elite batch's reward mean >= current overall elite buffer's reward mean
    if (e_reward_mean >= eB_reward_mean * adj_ratio):
        e_batch_buffer_msCEM.update_buffer(elite_batch)

    elite_ratio = 1 - unfit_count / len(msCEM_net_batch)

    # construct training data from elite_batch_buffer and global_eList
    elite_thetas = utils.convert_weight_thetas(e_batch_buffer_msCEM.get_single_batch())
    top_thetas = utils.convert_weight_thetas(global_eList_msCEM)

    # balance global elite models and elite buffer models count
    g_eB_target = 0.2
    mul_count = max(1, round(
        (g_eB_target * len(elite_thetas)) / (1 - g_eB_target) / len(top_thetas)))
    top_thetas = top_thetas * mul_count
    g_eB_ratio = len(top_thetas) / (len(top_thetas) + len(elite_thetas))
    elite_thetas.extend(top_thetas)

    # Update theta_mean, theta_std
    theta_mean_new = np.mean(elite_thetas, axis=0)
    theta_std_new = np.std(elite_thetas, axis=0)

    return e_batch_buffer_msCEM, global_eList_msCEM, theta_mean_new, theta_std_new, reward_m, ave_reward_std, reward_max, reward_accum, elite_ratio, e_reward_mean, global_eReward


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args, _ = load_args()
    print(args)

    LAYER = vars(args)['layer']
    LEARNING_RATE = vars(args)['lr']
    HIDDEN_SIZE = vars(args)['hidden_s']
    NOISE_SIZE = vars(args)['ze']
    WEIGHT_LATENT_SIZE = vars(args)['z']
    # BATCH_SIZE = vars(args)['ep_batch']  # number of episodes in a batch
    NET_BATCH = vars(args)['batch_size']  # number of guiding CEM models/episodes (batch size)
    PERCENTILE = vars(args)['percentile']
    use_cuda = args.use_cuda
    DEVICE = args.device
    # monitor_directory = 'monitor_CEM-AH'
    model_directory = 'model_CEM-AH'
    writer_directory = args.write_dir

    exp_name = 'MiniGrid-SimpleCrossingModS9N3-v0'
    COMMENT = args.comment
    SEED = 110

    # GAN settings
    gan_epochs = vars(args)['epochs']
    d_steps = 2000  # epoch for discriminator
    g_steps = 2000  # epoch for generator

    config = '_noise' + str(NOISE_SIZE) + '_weightZ' + str(WEIGHT_LATENT_SIZE) + '_percent' + str(
        PERCENTILE) + '_net' + str(NET_BATCH) + '_layer' + str(
        LAYER) + '_lr' + str(LEARNING_RATE)
    event_time = '_'+datetime.now().strftime("%Y%m%d_%H%M%S")

    # path = os.path.join(monitor_directory, exp_name+config+event_time)
    # if not os.path.exists(path):
    #     os.makedirs(path)

    writer_path = os.path.join(writer_directory, exp_name+config+event_time+"-"+COMMENT)
    if not os.path.exists(writer_path):
        os.makedirs(writer_path)

    model_path = os.path.join(model_directory, exp_name+config+event_time+"-"+COMMENT)
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    writer = SummaryWriter(writer_path)

    env = gym.make(exp_name)
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)  # Get rid of the 'mission' field
    env.set_max_steps(10000)
    env.seed(SEED)  # set seed to 110
    obs = env.reset()

    obs_size = obs.size
    n_actions = env.action_space.n

    print('Active CUDA Device: GPU', torch.cuda.current_device())
    print('Available devices ', torch.cuda.device_count())

    device = torch.device("cuda:" + DEVICE if torch.cuda.is_available() else "cpu")
    if not use_cuda:
        device = torch.device("cpu")
    print(device)

    # main policy network, includes encoder, W1 and W2 class
    if LAYER == 2:   # 2-layer network
        model = models.CEM_AH_2Layer(args, obs_size, n_actions)
        model = model.to(device)
        l1, b1 = model.get_gen_weights()
        weight_size = np.sum(list(l1.flatten().shape) + list(b1.flatten().shape))

    if LAYER == 3:  # 3-layer network
        model = models.CEM_AH_3Layer(args, obs_size, n_actions)
        model = model.to(device)
        l1, b1, l2, b2 = model.get_gen_weights()
        weight_size = np.sum(
            list(l1.flatten().shape) + list(b1.flatten().shape) + list(l2.flatten().shape) + list(b2.flatten().shape))

    dis = models.DiscriminatorZ(args, obs_size, n_actions)
    dis = dis.to(device)

    # ms-CEM policy (refers to the guiding CEM policy)
    dim_theta = (obs_size + 1) * n_actions
    # Initialize mean and standard deviation
    theta_mean = np.zeros(dim_theta)
    theta_std = np.ones(dim_theta)
    theta = np.random.multivariate_normal(theta_mean, np.diag(theta_std ** 2))
    if LAYER == 2:  # 2-layer network
        msCEM_model = models.NetW_2Layer(args, obs_size, n_actions)

    if LAYER == 3:  # 3-layer network
        msCEM_model = models.NetW_3Layer(args, obs_size, n_actions)
    msCEM_model = msCEM_model.to(device)

    criterionGen = nn.BCELoss()
    criterionDis = nn.BCELoss()
    criterionMSE = nn.MSELoss()
    print(model)
    print(dis)

    print("Observation Size: {} \t Action Size: {}".format(obs_size, n_actions))
    print("Model_param: {}".format(utils.count_parameters(model)))
    print("Encoder_param: {}".format(utils.count_parameters(model.encoder)))
    if LAYER == 2:  # 2-layer network
        print("W1_param: {} ".format(utils.count_parameters(model.W1)))
    if LAYER == 3:  # 3-layer network
        print("W1_param: {} \tW2_param: {}".format(utils.count_parameters(model.W1), utils.count_parameters(model.W2)))
    print("Discriminator_param: {}".format(utils.count_parameters(dis)))
    print("msCEM_param: {}".format(dim_theta))

    # print('======================== GAN Training ========================')
    start_time = time.time()

    # define agent start state
    agent_start_list = [((1, 1), 0)]

    # initialise all the elite buffers and global_eList (storing top elite across iterations)
    e_batch_buffer_msCEM = utils.elite_batch_bufferW_msCEM(100)  # 100 * (100 episodes) * (10k max steps) = 1e8
    global_eList_msCEM = []

    total_reward_accum = 0
    total_reward_accum_single = 0
    total_reward_accum_msCEM = 0

    test_reward_mean_dict = OrderedDict()
    test_reward_std_dict = OrderedDict()
    test_reward_max_dict = OrderedDict()
    test_reward_accum_dict = OrderedDict()
    test_step_counts_mean_dict = OrderedDict()
    test_step_counts_std_dict = OrderedDict()
    test_elite_ratio_dict = OrderedDict()

    for agent_start in agent_start_list:
        test_reward_mean_dict[agent_start] = 0
        test_reward_std_dict[agent_start] = 0
        test_reward_max_dict[agent_start] = 0
        test_reward_accum_dict[agent_start] = 0
        test_step_counts_mean_dict[agent_start] = 0
        test_step_counts_std_dict[agent_start] = 0
        test_elite_ratio_dict[agent_start] = 0

    optimE = optim.Adam(model.encoder.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimW1 = optim.Adam(model.W1.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    if LAYER == 3:  # 3-layer network
        optimW2 = optim.Adam(model.W2.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimDis = optim.Adam(dis.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    update_ratio = args.update_ratio
    update_ratio_decay_rate = args.update_decay

    gen_update = 1

    eval_net_batch = 10
    eval_ep_batch = 1

    for gan_epoch in range(gan_epochs):
        ratio = (gan_epoch+1)/(gan_epoch+2)

        # testing/eval with ensemble
        test_start = time.time()
        eval_ep_per_state = 10  # 10 episodes

        for state_num, agent_start in enumerate(agent_start_list):
            start_pos = agent_start[0]  # col, row
            start_dir = agent_start[1]
            env.set_initial_pos_dir(start_pos, start_dir)

            test_batch = []
            for episodes in range(eval_ep_per_state):
                test_batch.append(utils.sample_episodesNorm_eval(env, model, ensemble_size=10, use_cuda=use_cuda, device=device, seed=SEED))  # testing with ensemble size 10

            # get the test_batch stats
            test_reward_mean_dict[agent_start], test_reward_std_dict[agent_start], test_reward_max_dict[agent_start], test_reward_accum_dict[agent_start], test_step_counts_mean_dict[agent_start], test_step_counts_std_dict[agent_start], test_elite_ratio_dict[agent_start] = utils.get_batch_stats_eval(
                test_batch, PERCENTILE)

            elapsed_time = time.time() - start_time
            print('======================== State {} ========================'.format((state_num, len(agent_start_list))))
            print(
                "%d: test_reward_mean=%.5f, test_reward_std=%.5f, test_elite_ratio=%.3f, \tElapsed:%s" % (
                    gan_epoch, test_reward_mean_dict[agent_start], test_reward_std_dict[agent_start], test_elite_ratio_dict[agent_start],
                    time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

        test_duration = time.time() - test_start

        test_reward_mean = np.mean(list(test_reward_mean_dict.values()))
        test_reward_std = np.max(list(test_reward_std_dict.values()))
        test_reward_max = np.max(list(test_reward_max_dict.values()))
        test_reward_accum = np.sum(list(test_reward_accum_dict.values()))
        test_step_counts_mean = np.mean(list(test_step_counts_mean_dict.values()))
        test_step_counts_std = np.max(list(test_step_counts_std_dict.values()))
        test_elite_ratio = np.mean(list(test_elite_ratio_dict.values()))

        print('================================= Test Sample Complete \tDuration:%s ======================================='% (time.strftime("%H:%M:%S", time.gmtime(test_duration))))

        # sample episodes from current msCEM-policy and train a step
        print('======================== Sampling and training msCEM (1 step) ========================')
        train_start_cem = time.time()

        e_batch_buffer_msCEM, global_eList_msCEM, theta_mean, theta_std, reward_m_msCEM, ave_reward_std_msCEM, reward_max_msCEM, reward_accum_msCEM, elite_ratio_msCEM, e_reward_mean_msCEM, global_eReward_msCEM = sample_train_msCEM(
            e_batch_buffer_msCEM, global_eList_msCEM, env, msCEM_model, NET_BATCH, 1, theta_mean, theta_std, use_cuda, device, seed=SEED)

        train_duration_cem = time.time() - train_start_cem

        theta_std_mean = np.mean(theta_std)

        elapsed_time = time.time() - start_time
        print(
            "%d: reward_m_msCEM=%.5f, e_reward_mean_msCEM=%.5f, global_eReward_msCEM=%.3f, elite_ratio_msCEM=%.5f \tElapsed:%s" % (
                gan_epoch, reward_m_msCEM, e_reward_mean_msCEM, global_eReward_msCEM, elite_ratio_msCEM,
                time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

        # construct real gan samples
        CEM_batch = e_batch_buffer_msCEM.get_single_batch()
        CEM_obs_v, CEM_act_v, CEM_act_probs = utils.convert_weightbatch_obs_act(CEM_batch)
        all_CEM_obs_v_buffer = CEM_obs_v.reshape(CEM_obs_v.shape[0], -1)
        # standardize observations
        all_obs_v_buffer = (all_CEM_obs_v_buffer - all_CEM_obs_v_buffer.mean(dim=1, keepdim=True)) / all_CEM_obs_v_buffer.std(dim=1, keepdim=True)
        all_act_probs_v_buffer = CEM_act_probs

        CEM_obs_v_top, CEM_act_v_top, CEM_act_probs_top = utils.convert_weightbatch_obs_act(global_eList_msCEM)
        obs_v_top = CEM_obs_v_top.reshape(CEM_obs_v_top.shape[0], -1)

        # standardize observations
        all_obs_v_top = (obs_v_top - obs_v_top.mean(dim=1, keepdim=True)) / obs_v_top.std(dim=1, keepdim=True)
        all_act_probs_v_top = CEM_act_probs_top

        all_obs_v_buffer = torch.cat((all_obs_v_buffer, all_obs_v_top), 0)
        all_act_probs_v_buffer = torch.cat((all_act_probs_v_buffer, all_act_probs_v_top), 0)
        print('Total real tuples: {}'.format(all_obs_v_buffer.shape[0]))

        # reassign obs_v_buffer, act_probs_v_buffer (real/elite data) to remove computation graph
        all_obs_v_buffer = all_obs_v_buffer.cpu().data.numpy()
        all_obs_v_buffer = torch.from_numpy(all_obs_v_buffer)
        all_act_probs_v_buffer = all_act_probs_v_buffer.cpu().data.numpy()
        all_act_probs_v_buffer = torch.from_numpy(all_act_probs_v_buffer)
        if use_cuda:
            all_obs_v_buffer = all_obs_v_buffer.to(device)
            all_act_probs_v_buffer = all_act_probs_v_buffer.to(device)

        # Define pre-training batch size
        pretrain_batch = 500
        # Define discriminator batch size
        gan_batch = max(1, math.ceil(all_obs_v_buffer.shape[0] / 20))

        gen_obs_v, gen_act_probs_v, gen_targets, gen_dataloader = create_real_batch_loader(all_obs_v_buffer,
                                                                                           all_act_probs_v_buffer,
                                                                                           batch_size=pretrain_batch,
                                                                                           use_cuda=use_cuda,
                                                                                           device=device, shuffle=True)
        gen_targets_np = gen_targets.cpu().data.numpy()

        # pretrain generator
        criterionMSE = nn.MSELoss()
        optimE_gen = optim.Adam(model.encoder.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
        optimW1_gen = optim.Adam(model.W1.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
        if LAYER == 3:  # 3-layer network
            optimW2_gen = optim.Adam(model.W2.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

        train_start_pretrain = time.time()

        for pretrain_step in range(300):  # 1000
            # pretrain on shuffled real data
            model.encoder.train()
            model.W1.train()
            if LAYER == 3:  # 3-layer network
                model.W2.train()

            all_real_data = torch.zeros(0)
            all_fake_data = torch.zeros(0)
            if use_cuda:
                all_real_data = all_real_data.to(device)
                all_fake_data = all_fake_data.to(device)

            pretrain_count = 0
            for data in gen_dataloader:
                if data[0].shape[0] == 1:
                    continue
                optimE_gen.zero_grad()
                optimW1_gen.zero_grad()
                if LAYER == 3:  # 3-layer network
                    optimW2_gen.zero_grad()

                # # standardize data
                gen_real_obs = data[0]
                real_act_probs_v = data[1]
                all_real_data = torch.cat([all_real_data, real_act_probs_v], dim=0)

                # generate fake data
                obs_per_net = max(1, math.ceil(data[0].shape[0] / NET_BATCH))
                fake_obs_v, fake_act_probs_v = sample_fake_eObs_netBatch_mini(model, gen_real_obs, NET_BATCH,
                                                                              obs_per_net, obs_size, n_actions, use_cuda, device,
                                                                              show=False)

                # computation graph is maintained
                all_fake_data = torch.cat([all_fake_data, fake_act_probs_v], dim=0)

                pretrain_loss = criterionMSE(fake_act_probs_v, real_act_probs_v)
                pretrain_loss.backward()
                optimE_gen.step()
                optimW1_gen.step()
                if LAYER == 3:  # 3-layer network
                    optimW2_gen.step()

                pretrain_count += 1
                if pretrain_count >= 10:
                    break

            # evaluate on all data
            model.encoder.eval()
            model.W1.eval()
            if LAYER == 3:  # 3-layer network
                model.W2.eval()
            with torch.no_grad():
                real_data = all_real_data
                fake_data = all_fake_data

                pretrain_loss = criterionMSE(all_fake_data, all_real_data)
            elapsed_time = time.time() - start_time

            # after every N  iterations:
            if pretrain_step % 100 == 0:
                print("Iteration %d Pretrain Epoch %d: loss=%.5f, gan_tuple=%d \tElapsed:%s "
                      % (gan_epoch, pretrain_step, pretrain_loss.item(), all_real_data.shape[0],
                         time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

            if (pretrain_step == 0):
                min_pretrain_loss = pretrain_loss.item()

            # make a checkpoint of the pretrained model (whole model)
            if pretrain_loss.item() <= min_pretrain_loss:
                best_state_dict = model.state_dict()

        train_duration_pretrain = time.time() - train_start_pretrain

        print('============================= Pre-train generator completed =============================')

        # load the best_state_dict checkpoint
        model.load_state_dict(best_state_dict)

        print('==================== Discriminator and Generator (adversarial training)  =====================')

        dis_obs_v, dis_act_probs_v, dis_targets, dis_dataloader = create_real_batch_loader(all_obs_v_buffer,
                                                                                           all_act_probs_v_buffer,
                                                                                           batch_size=gan_batch,
                                                                                           use_cuda=use_cuda,
                                                                                           device=device, shuffle=True)
        dis_targets_np = dis_targets.cpu().data.numpy()

        all_real_input = torch.zeros(0, obs_size + n_actions)
        all_fake_input = torch.zeros(0, obs_size + n_actions)
        if use_cuda:
            all_real_input = all_real_input.to(device)
            all_fake_input = all_fake_input.to(device)

        train_start_gan = time.time()

        # For each batch in the dataloader
        for i, data in enumerate(dis_dataloader, 0):

            ########################
            # Update Discriminator
            ########################
            for j in range(update_ratio):
                dis.train()
                model.encoder.eval()
                model.W1.eval()
                if LAYER == 3:  # 3-layer network
                    model.W2.eval()
                # Train with all-real batch
                if data[0].shape[0] == 1:
                    continue
                dis.zero_grad()
                # add noise to data, zero mean and std ranges from 0.1 to 0, decaying with each mini-bath iteration
                noise_D = utils.create_d(data[0].shape[1] + data[1].shape[1])
                scale = min(0.1, 0.1 / (1 + gan_epoch) * 10)
                noise = utils.sample_d(noise_D, shape=data[0].shape[0], scale=scale)
                if use_cuda:
                    noise = noise.to(device)

                # standardize data
                dis_real_obs = data[0]
                # concat observation space with action probabilities
                real_obs_act_cat = torch.cat((dis_real_obs, data[1]), 1)
                dis_real_data = real_obs_act_cat + noise
                real_target = data[2]
                dis_real_out, _ = dis(dis_real_data)
                dis_loss_real = criterionDis(dis_real_out, real_target)
                # Calculate gradients for D in backward pass
                dis_loss_real.backward()
                D_x = dis_real_out.mean().item()

                # Train with all-fake batch
                # generate fake data
                obs_per_net = max(1, math.ceil(dis_real_obs.shape[0] / NET_BATCH))
                fake_obs_v, fake_act_probs_v = sample_fake_eObs_netBatch_mini(model, dis_real_obs, NET_BATCH,
                                                                              obs_per_net, obs_size, n_actions, use_cuda, device,
                                                                              show=False)

                fake_obs_v = fake_obs_v.cpu().data.numpy()
                fake_obs_v = torch.from_numpy(fake_obs_v)
                fake_act_probs_v = fake_act_probs_v.cpu().data.numpy()
                fake_act_probs_v = torch.from_numpy(fake_act_probs_v)

                if use_cuda:
                    fake_obs_v = fake_obs_v.to(device)
                    fake_act_probs_v = fake_act_probs_v.to(device)

                # concat observation space with action probabilities
                fake_obs_act_cat = torch.cat((fake_obs_v, fake_act_probs_v), 1)
                # add noise to data, zero mean and std ranges from 0.1 to 0, decaying with each mini-bath iteration
                noise_D = utils.create_d(fake_obs_act_cat.shape[1])
                scale = min(0.1, 0.1 / (1 + gan_epoch) * 10)
                noise = utils.sample_d(noise_D, shape=fake_obs_act_cat.shape[0], scale=scale)
                if use_cuda:
                    noise = noise.to(device)

                dis_fake_data = fake_obs_act_cat + noise
                fake_target = Variable(torch.ones(dis_fake_data.shape[0], 1)) * 0.1
                if use_cuda:
                    fake_target = fake_target.to(device)
                dis_fake_out, dis_fake_feature = dis(dis_fake_data)
                dis_loss_fake = criterionDis(dis_fake_out, fake_target)
                # Calculate gradients for D in backward pass
                dis_loss_fake.backward()
                D_G_z1 = dis_fake_out.mean().item()

                # Add the gradients from the all-real and all-fake batches
                dis_loss = dis_loss_real + dis_loss_fake
                # Update D
                optimDis.step()

            all_real_input = torch.cat((all_real_input, real_obs_act_cat), 0)
            all_fake_input = torch.cat((all_fake_input, fake_obs_act_cat), 0)

            ########################
            # Update Generator
            ########################
            for k in range(gen_update):
                dis.eval()
                model.encoder.train()
                model.W1.train()
                if LAYER == 3:  # 3-layer network
                    model.W2.train()
                model.encoder.zero_grad()
                model.W1.zero_grad()
                if LAYER == 3:  # 3-layer network
                    model.W2.zero_grad()

                gen_inputs = dis_fake_data
                gen_targets = Variable(torch.ones(gen_inputs.shape[0], 1))
                if use_cuda:
                    gen_targets = gen_targets.to(device)
                # Since we just updated D, perform another forward pass of all-fake batch through D
                gen_out, layer_f = dis(gen_inputs)  # debug with feature_loss
                _, layer_r = dis(dis_real_data)  # debug with feature_loss

                # for mse loss
                gen_fake_act_probs_v = dis_fake_data[:, -n_actions:]  # get the last few elements (actions)
                gen_real_act_probs_v = dis_real_data[:, -n_actions:]  # get the last few elements (actions)

                # Calculate G's loss based on this output
                # feature_loss = criterionMSE(layer_f, layer_r.detach())
                mse_loss = criterionMSE(gen_fake_act_probs_v, gen_real_act_probs_v)
                fake_loss = criterionGen(gen_out, gen_targets)
                gen_loss = mse_loss + fake_loss
                # Calculate gradients for G
                gen_loss.backward()
                D_G_z2 = gen_out.mean().item()
                # Update G
                optimE.step()
                optimW1.step()
                if LAYER == 3:  # 3-layer network
                    optimW2.step()

            # Output training stats
            print('[%d/%d][%d/%d]\t\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (gan_epoch, gan_epochs, i, len(dis_dataloader),
                     dis_loss.item(), gen_loss.item(), D_x, D_G_z1, D_G_z2))

            batch_count = gan_epoch * len(dis_dataloader) + i
            writer.add_scalar("miniBatchUpdate/Discriminator_loss", dis_loss.item(), batch_count)
            writer.add_scalar("miniBatchUpdate/Generator_loss", gen_loss.item(), batch_count)
            writer.add_scalar("miniBatchUpdate/D(x)", D_x, batch_count)
            writer.add_scalar("miniBatchUpdate/D(G(z1))", D_G_z1, batch_count)
            writer.add_scalar("miniBatchUpdate/D(G(z2))", D_G_z2, batch_count)

        train_duration_gan = time.time() - train_start_gan

        # evaluate on all data (real and fake)
        dis.eval()
        model.encoder.eval()
        model.W1.eval()
        if LAYER == 3:  # 3-layer network
            model.W2.eval()
        with torch.no_grad():
            # dis real data
            dis_out, _ = dis(all_real_input)
            pred_prob = dis_out.cpu().data.numpy()  # these

            dis_targets = Variable(torch.ones(all_real_input.shape[0], 1))
            dis_targets = dis_targets.to(device)

            dis_loss_real = criterionDis(dis_out, dis_targets)
            dis_class = np.zeros(pred_prob.shape)
            dis_class[pred_prob > 0.5] = 1
            # compare with targets
            dis_targets_np[dis_targets_np <= 0.5] = 0  # change the soft labels back to hard labels
            dis_targets_np[dis_targets_np > 0.5] = 1
            correct_real = np.sum(dis_class == dis_targets_np)

            # dis fake data
            dis_out_fake, _ = dis(all_fake_input)
            pred_prob_fake = dis_out_fake.cpu().data.numpy()  # these
            dis_targets_fake = Variable(torch.zeros(all_fake_input.shape[0], 1))
            if use_cuda:
                dis_targets_fake = dis_targets_fake.to(device)
            dis_loss_fake = criterionDis(dis_out_fake, dis_targets_fake)
            dis_class = np.zeros(pred_prob_fake.shape)
            dis_class[pred_prob_fake > 0.5] = 1
            # compare with targets
            dis_targets_np_fake = dis_targets_fake.cpu().data.numpy()
            dis_targets_np_fake[dis_targets_np_fake <= 0.5] = 0  # change the soft labels back to hard labels
            dis_targets_np_fake[dis_targets_np_fake > 0.5] = 1
            correct_fake = np.sum(dis_class == dis_targets_np_fake)

            # gen fake data
            gen_inputs = all_fake_input
            gen_targets = Variable(torch.ones(gen_inputs.shape[0], 1))
            if use_cuda:
                gen_targets = gen_targets.to(device)
            # Since we just updated D, perform another forward pass of all-fake batch through D
            gen_out, layer_f = dis(gen_inputs)  # debug with feature_loss
            _, layer_r = dis(all_real_input)  # debug with feature_loss

            # for mse loss
            gen_fake_act_probs_v = dis_fake_data[:, -n_actions:]  # get the last few elements (actions)
            gen_real_act_probs_v = dis_real_data[:, -n_actions:]  # get the last few elements (actions)

            # Calculate G's loss based on this output
            # feature_loss = criterionMSE(layer_f, layer_r.detach())
            mse_loss = criterionMSE(gen_fake_act_probs_v, gen_real_act_probs_v)
            fake_loss = criterionGen(gen_out, gen_targets)
            gen_loss = mse_loss + fake_loss
            fake_success = (gen_out > 0.5).sum().item()

        elapsed_time = time.time() - start_time
        print('========================================================================')
        print("Iteration %d, d_real_loss=%.5f, d_fake_loss=%.5f, dis_real_correct=%.5f, dis_fake_correct=%.5f, "
              "fake_loss=%.5f, gen_loss=%.5f, fake_success=%.5f, gan_tuple=%d, update_ratio=%d, gen_update=%d\tElapsed:%s "
              % (gan_epoch, dis_loss_real.item(), dis_loss_fake.item(), correct_real / all_real_input.shape[0],
                 correct_fake / all_fake_input.shape[0],
                 fake_loss.item(), gen_loss.item(), fake_success / all_fake_input.shape[0],
                 all_real_input.shape[0], update_ratio, gen_update,
                 time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
        print('========================================================================')

        d_real_accuracy = (correct_real / all_real_input.shape[0])
        d_fake_accuracy = (correct_fake / all_fake_input.shape[0])
        d_real_loss = dis_loss_real.item()
        d_fake_loss = dis_loss_fake.item()

        g_accuracy = (fake_success / all_fake_input.shape[0])

        total_reward_accum += test_reward_accum
        total_reward_accum_msCEM += reward_accum_msCEM

        total_train_duration = train_duration_cem + train_duration_pretrain + train_duration_gan

        # write results using tensorboard
        writer.add_scalar("loss/Discriminator_real_loss", d_real_loss, gan_epoch)
        writer.add_scalar("loss/Discriminator_fake_loss", d_fake_loss, gan_epoch)
        writer.add_scalar("loss/Generator_loss", gen_loss.item(), gan_epoch)
        writer.add_scalar("loss/MSE_loss", mse_loss.item(), gan_epoch)
        writer.add_scalar("loss/Fake_loss", fake_loss.item(), gan_epoch)
        writer.add_scalar("loss/D_G_update", update_ratio, gan_epoch)
        writer.add_scalar("accuracy/Discriminator_real_acc", d_real_accuracy, gan_epoch)
        writer.add_scalar("accuracy/Discriminator_fake_acc", d_fake_accuracy, gan_epoch)
        writer.add_scalar("accuracy/Generator_acc", g_accuracy, gan_epoch)

        writer.add_scalar("reward/reward_max", test_reward_max, gan_epoch)
        writer.add_scalar("reward/reward_accum", total_reward_accum, gan_epoch)
        writer.add_scalar("reward/reward_mean", test_reward_mean, gan_epoch)
        writer.add_scalar("reward/reward_std", test_reward_std, gan_epoch)
        writer.add_scalar("step_count/step_counts_mean", test_step_counts_mean, gan_epoch)
        writer.add_scalar("step_count/step_counts_std", test_step_counts_std, gan_epoch)
        writer.add_scalar("elite/elite_ratio", test_elite_ratio, gan_epoch)

        writer.add_scalar("msCEM/reward_max", reward_max_msCEM, gan_epoch)
        writer.add_scalar("msCEM/reward_accum", total_reward_accum_msCEM, gan_epoch)
        writer.add_scalar("msCEM/reward_mean", reward_m_msCEM, gan_epoch)
        writer.add_scalar("msCEM/ave_reward_std", ave_reward_std_msCEM, gan_epoch)
        writer.add_scalar("msCEM/elite_ratio", elite_ratio_msCEM, gan_epoch)
        writer.add_scalar("msCEM/elite_reward_mean", e_reward_mean_msCEM, gan_epoch)
        writer.add_scalar("msCEM/global_ep_counts", len(global_eList_msCEM), gan_epoch)
        writer.add_scalar("msCEM/global_reward", global_eReward_msCEM, gan_epoch)
        writer.add_scalar("msCEM/theta_std_mean", theta_std_mean, gan_epoch)

        writer.add_scalar("duration/train_duration_cem", train_duration_cem, gan_epoch)
        writer.add_scalar("duration/train_duration_pretrain", train_duration_pretrain, gan_epoch)
        writer.add_scalar("duration/train_duration_gan", train_duration_gan, gan_epoch)
        writer.add_scalar("duration/total_train_duration", total_train_duration, gan_epoch)

        writer.add_scalar("duration/test_duration", test_duration, gan_epoch)
        writer.flush()

        # schedule for discriminator
        update_ratio = max(1, int(update_ratio * (1 - update_ratio_decay_rate)))

        # schedule for generator
        if update_ratio <= 5:
            gen_update = min(args.update_ratio, gen_update + 1)

        # save model
        torch.save(model.state_dict(), model_path + '_main_net' + '.pth')
        # save discriminator
        torch.save(dis.state_dict(), model_path + '_dis' + '.pth')

        # save the theta_mean and theta_std of CEM
        with open(model_path + '_CEM' +'.pkl', 'wb') as f:
            pickle.dump((theta_mean, theta_std), f, protocol=pickle.HIGHEST_PROTOCOL)

    writer.close()
    env.close()
