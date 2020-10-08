import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
from scipy.special import softmax

import utils


class CEM_AH_2Layer(nn.Module):
    def __init__(self, args, obs_size, n_actions):
        super(CEM_AH_2Layer, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.gpu = torch.device("cuda:" + self.device if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.encoder = EncoderW(args)
        self.W1 = GeneratorW1B_2Layer(args, obs_size, n_actions)
        self.tanh = nn.Tanh()
        self.sm = nn.Softmax(dim=1)
        self.x_dist = utils.create_d(self.ze)  # encoder dimension - create_d: create the distribution (multivaraite normal)
        self.sample_noise()

    def forward(self, x):
        # get the layer weights from noise z
        z = self.z
        codes = self.encoder(z)
        if z.shape[0] == 1:
            l1_b1 = self.W1(codes)  # codes[0]: encoded info for layer 1, l1: main net layer1 weights
            l1, b1 = l1_b1[0][0], l1_b1[1][0]
        else:
            l1_b1 = self.W1(codes)  # codes[0]: encoded info for layer 1, l1: main net layer1 weights
            l1, b1 = l1_b1[0], l1_b1[1]
        x = F.linear(x, l1) + b1
        x = self.tanh(x)
        x = self.sm(x)
        return x

    def ensemble_forward(self, x, noise_set_count):
        noise_set = self.sample_noise_set(noise_set_count)
        codes = self.encoder(noise_set)
        l1_b1 = self.W1(codes)  # codes[0]: encoded info for layer 1, l1: main net layer1 weights
        l1, b1 = l1_b1[0], l1_b1[1]

        l1_concat = l1.reshape(-1, l1.shape[2])
        b1_concat = b1.flatten()

        x = F.linear(x, l1_concat) + b1_concat
        x = self.tanh(x)
        x = x.reshape(l1.shape[0], -1)
        x = self.sm(x)
        return x

    def replace_forward(self, x, l1, b1):

        x = F.linear(x, l1) + b1
        x = self.tanh(x)
        x = self.sm(x)
        return x

    def sample_noise(self):
        self.z = utils.sample_d(self.x_dist, 1, use_cuda=self.use_cuda, device=self.gpu)

    def sample_noise_set(self, noise_set_count):
        return utils.sample_d(self.x_dist, noise_set_count, use_cuda=self.use_cuda, device=self.gpu)

    def set_noise(self, z):
        self.z = z

    def get_gen_weights(self):
        codes = self.encoder(self.z)
        if self.z.shape[0] == 1:
            l1_b1 = self.W1(codes)  # codes[0]: encoded info for layer 1, l1: main net layer1 weights
            self.l1, self.b1 = l1_b1[0][0], l1_b1[1][0]
        else :
            l1_b1 = self.W1(codes)  # codes[0]: encoded info for layer 1, l1: main net layer1 weights
            self.l1, self.b1 = l1_b1[0], l1_b1[1]
        return self.l1, self.b1

    def sample_ensemble_weights(self, noise_set_count):
        noise_set = self.sample_noise_set(noise_set_count)
        codes = self.encoder(noise_set)
        l1_b1 = self.W1(codes)  # codes[0]: encoded info for layer 1, l1: main net layer1 weights
        l1, b1 = l1_b1[0], l1_b1[1]
        l1b1 = utils.concat_layer_bias(l1, b1)
        return l1b1


class CEM_AH_3Layer(nn.Module):
    def __init__(self, args, obs_size, n_actions):
        super(CEM_AH_3Layer, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.gpu = torch.device("cuda:" + self.device if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.encoder = EncoderW(args)
        self.W1 = GeneratorW1B(args, obs_size)
        self.W2 = GeneratorW2B(args, n_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sm = nn.Softmax(dim=1)
        self.x_dist = utils.create_d(self.ze)  # encoder dimension - create_d: create the distribution (multivaraite normal)
        self.sample_noise()

    def forward(self, x):
        # get the layer weights from noise z
        z = self.z
        codes = self.encoder(z)
        if z.shape[0] == 1:
            l1_b1 = self.W1(codes)  # codes[0]: encoded info for layer 1, l1: main net layer1 weights
            l1, b1 = l1_b1[0][0], l1_b1[1][0]
            l2_b2 = self.W2(codes)  # codes[1]: encoded info for layer 2, l2: main net layer2 weights
            l2, b2 = l2_b2[0][0], l2_b2[1][0]
        else:
            l1_b1 = self.W1(codes)  # codes[0]: encoded info for layer 1, l1: main net layer1 weights
            l1, b1 = l1_b1[0], l1_b1[1]
            l2_b2 = self.W2(codes)  # codes[1]: encoded info for layer 2, l2: main net layer2 weights
            l2, b2 = l2_b2[0], l2_b2[1]

        x = F.linear(x, l1) + b1
        x = self.relu(x)
        x = F.linear(x, l2) + b2
        x = self.tanh(x)
        x = self.sm(x)
        return x

    def ensemble_forward(self, x, noise_set_count):
        noise_set = self.sample_noise_set(noise_set_count)
        codes = self.encoder(noise_set)
        l1_b1 = self.W1(codes)  # codes[0]: encoded info for layer 1, l1: main net layer1 weights
        l1, b1 = l1_b1[0], l1_b1[1]
        l2_b2 = self.W2(codes)  # codes[1]: encoded info for layer 2, l2: main net layer2 weights
        l2, b2 = l2_b2[0], l2_b2[1]

        l1_concat = l1.reshape(-1, l1.shape[2])
        b1_concat = b1.flatten()
        b2_concat = b2.flatten()
        l2_concat = l2.reshape(b2_concat.shape[0], -1)

        x = F.linear(x, l1_concat) + b1_concat
        x = self.relu(x)
        x = x.reshape(-1, l1.shape[1])
        x = F.linear(x, l2_concat) + b2_concat
        x = self.tanh(x)

        mask = utils.block_diag(torch.ones(x.shape[0], 1, self.n_actions)).to(self.gpu)
        x = x * mask
        x = torch.sum(x, dim=0)
        x = x.reshape(-1, self.n_actions)
        x = self.sm(x)
        return x

    def replace_forward(self, x, l1, b1, l2, b2):

        x = F.linear(x, l1) + b1
        x = self.relu(x)
        x = F.linear(x, l2) + b2
        x = self.tanh(x)
        x = self.sm(x)
        return x

    def sample_noise(self):
        self.z = utils.sample_d(self.x_dist, 1, use_cuda=self.use_cuda, device=self.gpu)

    def sample_noise_set(self, noise_set_count):
        return utils.sample_d(self.x_dist, noise_set_count, use_cuda=self.use_cuda, device=self.gpu)

    def set_noise(self, z):
        self.z = z

    def get_gen_weights(self):
        codes = self.encoder(self.z)
        if self.z.shape[0] == 1:
            l1_b1 = self.W1(codes)  # codes[0]: encoded info for layer 1, l1: main net layer1 weights
            self.l1, self.b1 = l1_b1[0][0], l1_b1[1][0]
            l2_b2 = self.W2(codes)  # codes[1]: encoded info for layer 2, l2: main net layer2 weights
            self.l2, self.b2 = l2_b2[0][0], l2_b2[1][0]
        else :
            l1_b1 = self.W1(codes)  # codes[0]: encoded info for layer 1, l1: main net layer1 weights
            self.l1, self.b1 = l1_b1[0], l1_b1[1]
            l2_b2 = self.W2(codes)  # codes[1]: encoded info for layer 2, l2: main net layer2 weights
            self.l2, self.b2 = l2_b2[0], l2_b2[1]
        return self.l1, self.b1, self.l2, self.b2


class CEM_AH_2Layer_continuous(nn.Module):
    def __init__(self, args, obs_size, n_actions):
        super(CEM_AH_2Layer_continuous, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.gpu = torch.device("cuda:" + self.device if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.encoder = EncoderW(args)
        self.W1 = GeneratorW1B_2Layer(args, obs_size, n_actions)
        self.tanh = nn.Tanh()
        self.x_dist = utils.create_d(self.ze)  # encoder dimension - create_d: create the distribution (multivaraite normal)
        self.sample_noise()

    def forward(self, x):
        # get the layer weights from noise z
        z = self.z
        codes = self.encoder(z)
        if z.shape[0] == 1:
            l1_b1 = self.W1(codes)  # codes[0]: encoded info for layer 1, l1: main net layer1 weights
            l1, b1 = l1_b1[0][0], l1_b1[1][0]
        else:
            l1_b1 = self.W1(codes)  # codes[0]: encoded info for layer 1, l1: main net layer1 weights
            l1, b1 = l1_b1[0], l1_b1[1]
        x = F.linear(x, l1) + b1
        x = self.tanh(x)
        return x

    def ensemble_forward(self, x, noise_set_count):
        noise_set = self.sample_noise_set(noise_set_count)
        codes = self.encoder(noise_set)
        l1_b1 = self.W1(codes)  # codes[0]: encoded info for layer 1, l1: main net layer1 weights
        l1, b1 = l1_b1[0], l1_b1[1]

        l1_concat = l1.reshape(-1, l1.shape[2])
        b1_concat = b1.flatten()

        x = F.linear(x, l1_concat) + b1_concat
        x = self.tanh(x)
        x = x.reshape(l1.shape[0], -1)
        return x

    def replace_forward(self, x, l1, b1):

        x = F.linear(x, l1) + b1
        x = self.tanh(x)
        return x

    def sample_noise(self):
        self.z = utils.sample_d(self.x_dist, 1, use_cuda=self.use_cuda, device=self.gpu)

    def sample_noise_set(self, noise_set_count):
        return utils.sample_d(self.x_dist, noise_set_count, use_cuda=self.use_cuda, device=self.gpu)

    def set_noise(self, z):
        self.z = z

    def get_gen_weights(self):
        codes = self.encoder(self.z)
        if self.z.shape[0] == 1:
            l1_b1 = self.W1(codes)  # codes[0]: encoded info for layer 1, l1: main net layer1 weights
            self.l1, self.b1 = l1_b1[0][0], l1_b1[1][0]
        else :
            l1_b1 = self.W1(codes)  # codes[0]: encoded info for layer 1, l1: main net layer1 weights
            self.l1, self.b1 = l1_b1[0], l1_b1[1]
        return self.l1, self.b1

    def sample_ensemble_weights(self, noise_set_count):
        noise_set = self.sample_noise_set(noise_set_count)
        codes = self.encoder(noise_set)
        l1_b1 = self.W1(codes)  # codes[0]: encoded info for layer 1, l1: main net layer1 weights
        l1, b1 = l1_b1[0], l1_b1[1]
        l1b1 = utils.concat_layer_bias(l1, b1)
        return l1b1


class CEM_AH_3Layer_continuous(nn.Module):
    def __init__(self, args, obs_size, n_actions):
        super(CEM_AH_3Layer_continuous, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.gpu = torch.device("cuda:" + self.device if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.encoder = EncoderW(args)
        self.W1 = GeneratorW1B(args, obs_size)
        self.W2 = GeneratorW2B(args, n_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.x_dist = utils.create_d(self.ze)  # encoder dimension - create_d: create the distribution (multivaraite normal)
        self.sample_noise()

    def forward(self, x):
        # get the layer weights from noise z
        z = self.z
        codes = self.encoder(z)
        if z.shape[0] == 1:
            l1_b1 = self.W1(codes)  # codes[0]: encoded info for layer 1, l1: main net layer1 weights
            l1, b1 = l1_b1[0][0], l1_b1[1][0]
            l2_b2 = self.W2(codes)  # codes[1]: encoded info for layer 2, l2: main net layer2 weights
            l2, b2 = l2_b2[0][0], l2_b2[1][0]
        else:
            l1_b1 = self.W1(codes)  # codes[0]: encoded info for layer 1, l1: main net layer1 weights
            l1, b1 = l1_b1[0], l1_b1[1]
            l2_b2 = self.W2(codes)  # codes[1]: encoded info for layer 2, l2: main net layer2 weights
            l2, b2 = l2_b2[0], l2_b2[1]

        x = F.linear(x, l1) + b1
        x = self.relu(x)
        x = F.linear(x, l2) + b2
        x = self.tanh(x)
        return x

    def ensemble_forward(self, x, noise_set_count):
        noise_set = self.sample_noise_set(noise_set_count)
        codes = self.encoder(noise_set)
        l1_b1 = self.W1(codes)  # codes[0]: encoded info for layer 1, l1: main net layer1 weights
        l1, b1 = l1_b1[0], l1_b1[1]
        l2_b2 = self.W2(codes)  # codes[1]: encoded info for layer 2, l2: main net layer2 weights
        l2, b2 = l2_b2[0], l2_b2[1]

        l1_concat = l1.reshape(-1, l1.shape[2])
        b1_concat = b1.flatten()
        b2_concat = b2.flatten()
        l2_concat = l2.reshape(b2_concat.shape[0], -1)

        x = F.linear(x, l1_concat) + b1_concat
        x = self.relu(x)
        x = x.reshape(-1, l1.shape[1])
        x = F.linear(x, l2_concat) + b2_concat
        x = self.tanh(x)

        mask = utils.block_diag(torch.ones(x.shape[0], 1, self.n_actions)).to(self.gpu)
        x = x * mask
        x = torch.sum(x, dim=0)
        x = x.reshape(-1, self.n_actions)
        return x

    def replace_forward(self, x, l1, b1, l2, b2):

        x = F.linear(x, l1) + b1
        x = self.relu(x)
        x = F.linear(x, l2) + b2
        x = self.tanh(x)
        return x

    def sample_noise(self):
        self.z = utils.sample_d(self.x_dist, 1, use_cuda=self.use_cuda, device=self.gpu)

    def sample_noise_set(self, noise_set_count):
        return utils.sample_d(self.x_dist, noise_set_count, use_cuda=self.use_cuda, device=self.gpu)

    def set_noise(self, z):
        self.z = z

    def get_gen_weights(self):
        codes = self.encoder(self.z)
        if self.z.shape[0] == 1:
            l1_b1 = self.W1(codes)  # codes[0]: encoded info for layer 1, l1: main net layer1 weights
            self.l1, self.b1 = l1_b1[0][0], l1_b1[1][0]
            l2_b2 = self.W2(codes)  # codes[1]: encoded info for layer 2, l2: main net layer2 weights
            self.l2, self.b2 = l2_b2[0][0], l2_b2[1][0]
        else :
            l1_b1 = self.W1(codes)  # codes[0]: encoded info for layer 1, l1: main net layer1 weights
            self.l1, self.b1 = l1_b1[0], l1_b1[1]
            l2_b2 = self.W2(codes)  # codes[1]: encoded info for layer 2, l2: main net layer2 weights
            self.l2, self.b2 = l2_b2[0], l2_b2[1]
        return self.l1, self.b1, self.l2, self.b2


class NetW_2Layer(nn.Module):
    def __init__(self, args, obs_size, n_actions):
        super(NetW_2Layer, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.linear1 = nn.Linear(obs_size, n_actions)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.tanh(x)
        return x


class NetW_3Layer(nn.Module):
    def __init__(self, args, obs_size, n_actions):
        super(NetW_3Layer, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.linear1 = nn.Linear(obs_size, self.hidden_s)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.hidden_s, n_actions)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.tanh(x)
        return x


class NetW_4Layer(nn.Module):
    def __init__(self, args, obs_size, n_actions):
        super(NetW_4Layer, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.linear1 = nn.Linear(obs_size, 400)  # self.hidden_s
        self.linear2 = nn.Linear(400, 300)
        self.linear3 = nn.Linear(300, n_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.tanh(x)
        return x


class CEM_QtOpt(nn.Module):
    def __init__(self, n_actions, action_lim):
        super(CEM_QtOpt, self).__init__()
        self.n_actions = n_actions
        self.action_lim = action_lim
        self.dim_theta = self.n_actions
        self.theta_mean = np.zeros(self.dim_theta).flatten()
        self.theta_std = np.ones(self.dim_theta).flatten()

    def initialize(self):
        self.theta_mean = np.zeros(self.dim_theta).flatten()
        self.theta_std = np.ones(self.dim_theta).flatten()

    def sample(self):
        theta = np.random.multivariate_normal(self.theta_mean, np.diag(self.theta_std ** 2))
        return theta

    def sample_multi(self, n):
        self.theta_list = np.vstack(
            [np.random.multivariate_normal(self.theta_mean, np.diag(self.theta_std ** 2)) for _ in range(n)])
        return self.theta_list

    def update(self, elite_thetas):
        self.theta_mean = np.mean(elite_thetas, axis=0).flatten()
        self.theta_std = np.std(elite_thetas, axis=0).flatten()
        return self.theta_mean, self.theta_std


class EncoderW(nn.Module):
    def __init__(self, args):
        super(EncoderW, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'Encoder'
        self.linear1 = nn.Linear(self.ze, 150)
        self.linear2 = nn.Linear(150, self.z)
        self.bn1 = nn.BatchNorm1d(150)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.reshape(-1, self.ze)  # flatten filter size
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class GeneratorW1B(nn.Module):
    def __init__(self, args, obs_size):
        super(GeneratorW1B, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW1'
        self.obs_size = obs_size
        self.layer1_size = args.hidden_s
        self.linear1 = nn.Linear(self.z, 200)  # 200
        self.linear2 = nn.Linear(200, self.obs_size * self.layer1_size)
        self.bias1 = nn.Linear(self.z, 200)
        self.bias2 = nn.Linear(200, self.layer1_size)
        self.bn1 = nn.BatchNorm1d(200)
        self.relu = nn.ReLU()

    def forward(self, x):
        weights = self.linear1(x)
        weights = self.relu(weights)

        weights = self.linear2(weights)
        weights = weights.view(-1, self.layer1_size, self.obs_size)

        bias = self.bias1(x)
        bias = self.relu(bias)
        bias = self.bias2(bias)
        bias = bias.view(-1, self.layer1_size)
        return weights, bias


class GeneratorW1B_2Layer(nn.Module):
    def __init__(self, args, obs_size, n_actions):
        super(GeneratorW1B_2Layer, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW1'
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.linear1 = nn.Linear(self.z, 200)  # 200
        self.linear2 = nn.Linear(200, self.obs_size * self.n_actions)
        self.bias1 = nn.Linear(self.z, 200)
        self.bias2 = nn.Linear(200, self.n_actions)
        self.bn1 = nn.BatchNorm1d(200)
        self.relu = nn.ReLU()

    def forward(self, x):
        weights = self.linear1(x)
        weights = self.relu(weights)

        weights = self.linear2(weights)
        weights = weights.view(-1, self.n_actions, self.obs_size)

        bias = self.bias1(x)
        bias = self.relu(bias)
        bias = self.bias2(bias)
        bias = bias.view(-1, self.n_actions)
        return weights, bias


class GeneratorW2B(nn.Module):
    def __init__(self, args, n_actions):
        super(GeneratorW2B, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW2'
        self.n_actions = n_actions
        self.layer2_size = args.hidden_s  # needs to be equal to layer1_size
        self.linear1 = nn.Linear(self.z, 100)  # 100
        self.linear2 = nn.Linear(100, self.layer2_size * self.n_actions)
        self.bias1 = nn.Linear(self.z, 100)
        self.bias2 = nn.Linear(100, self.n_actions)
        self.bn1 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()

    def forward(self, x):
        weights = self.linear1(x)
        weights = self.relu(weights)

        weights = self.linear2(weights)
        weights = weights.view(-1, self.n_actions, self.layer2_size)

        bias = self.bias1(x)
        bias = self.relu(bias)
        bias = self.bias2(bias)
        bias = bias.view(-1, self.n_actions)
        return weights, bias


class DiscriminatorZ(nn.Module):
    def __init__(self, args, obs_size, n_actions):
        super(DiscriminatorZ, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)

        self.name = 'DiscriminatorZ'
        self.obs_size = obs_size
        self.n_actions = n_actions
        out = 1  # output single value
        self.linear1 = nn.Linear(self.obs_size + self.n_actions, 200)
        self.linear2 = nn.Linear(200, 100)
        self.linear3 = nn.Linear(100, out)
        self.relu = nn.ELU()
        # self.lrelu = nn.LeakyReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(200)
        self.bn2 = nn.BatchNorm1d(100)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print ('Dz in: ', x.shape)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.bn2(x)
        features = self.relu(x)
        x = self.dropout2(features)
        x = self.linear3(x)
        logits = self.sigmoid(x)
        return logits, features


def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


DDPG_EPS = 0.003


class Critic_DDPG_2Layer(nn.Module):

    def __init__(self, args, state_dim, action_dim):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of input action (int)
        :return:
        """
        super(Critic_DDPG_2Layer, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fcs1 = nn.Linear(state_dim, self.hidden_s)
        self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())

        self.fca1 = nn.Linear(action_dim, self.hidden_s)
        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

        self.fc3 = nn.Linear(self.hidden_s*2, 1)
        self.fc3.weight.data.uniform_(-DDPG_EPS,DDPG_EPS)

    def forward(self, state, action):
        """
        returns Value function Q(s,a) obtained from critic network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param action: Input Action (Torch Variable : [n,action_dim] )
        :return: Value function : Q(S,a) (Torch Variable : [n,1] )
        """
        s1 = F.relu(self.fcs1(state))
        a1 = F.relu(self.fca1(action))
        x = torch.cat((s1, a1), dim=1)

        x = self.fc3(x)

        return x


class Actor_DDPG_2Layer(nn.Module):

    def __init__(self, args, state_dim, action_dim, action_lim):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of output action (int)
        :param action_lim: Used to limit action in [-action_lim,action_lim]
        :return:
        """
        super(Actor_DDPG_2Layer, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim

        self.fc1 = nn.Linear(state_dim, action_dim)
        self.fc1.weight.data.uniform_(-DDPG_EPS,DDPG_EPS)
        self.tanh = nn.Tanh()

    def forward(self, state):
        """
        returns policy function Pi(s) obtained from actor network
        this function is a gaussian prob distribution for all actions
        with mean lying in (-1,1) and sigma lying in (0,1)
        The sampled action can , then later be rescaled
        :param state: Input state (Torch Variable : [n,state_dim] )
        :return: Output action (Torch Variable: [n,action_dim] )
        """
        x = self.fc1(state)
        action = self.tanh(x)

        action = action.cpu().data.numpy() * self.action_lim
        action = torch.FloatTensor(action)

        return action


class Critic_DDPG_3Layer(nn.Module):

    def __init__(self, args, state_dim, action_dim):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of input action (int)
        :return:
        """
        super(Critic_DDPG_3Layer, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fcs1 = nn.Linear(state_dim, self.hidden_s)
        self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
        self.fcs2 = nn.Linear(self.hidden_s, self.hidden_s)
        self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())

        self.fca1 = nn.Linear(action_dim, self.hidden_s)
        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

        self.fc2 = nn.Linear(self.hidden_s*2, self.hidden_s)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(self.hidden_s, 1)
        self.fc3.weight.data.uniform_(-DDPG_EPS,DDPG_EPS)

    def forward(self, state, action):
        """
        returns Value function Q(s,a) obtained from critic network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param action: Input Action (Torch Variable : [n,action_dim] )
        :return: Value function : Q(S,a) (Torch Variable : [n,1] )
        """
        s1 = F.relu(self.fcs1(state))
        s2 = F.relu(self.fcs2(s1))

        a1 = F.relu(self.fca1(action))

        x = torch.cat((s2, a1), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Actor_DDPG_3Layer(nn.Module):

    def __init__(self, args, state_dim, action_dim, action_lim):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of output action (int)
        :param action_lim: Used to limit action in [-action_lim,action_lim]
        :return:
        """
        super(Actor_DDPG_3Layer, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim

        self.fc1 = nn.Linear(state_dim, self.hidden_s)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(self.hidden_s, action_dim)
        self.fc2.weight.data.uniform_(-DDPG_EPS,DDPG_EPS)
        self.tanh = nn.Tanh()

    def forward(self, state):
        """
        returns policy function Pi(s) obtained from actor network
        this function is a gaussian prob distribution for all actions
        with mean lying in (-1,1) and sigma lying in (0,1)
        The sampled action can , then later be rescaled
        :param state: Input state (Torch Variable : [n,state_dim] )
        :return: Output action (Torch Variable: [n,action_dim] )
        """
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        action = self.tanh(x)

        action = action.cpu().data.numpy() * self.action_lim
        action = torch.FloatTensor(action)

        return action


class Critic_DDPG_4Layer(nn.Module):

    def __init__(self, args, state_dim, action_dim):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of input action (int)
        :return:
        """
        super(Critic_DDPG_4Layer, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim + action_dim, 400)  # self.hidden_s
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(400, 300)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(300, 1)
        self.fc3.weight.data.uniform_(-DDPG_EPS, DDPG_EPS)

    def forward(self, state, action):
        """
        returns Value function Q(s,a) obtained from critic network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param action: Input Action (Torch Variable : [n,action_dim] )
        :return: Value function : Q(S,a) (Torch Variable : [n,1] )
        """
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Actor_DDPG_4Layer(nn.Module):

    def __init__(self, args, state_dim, action_dim, action_lim):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of output action (int)
        :param action_lim: Used to limit action in [-action_lim,action_lim]
        :return:
        """
        super(Actor_DDPG_4Layer, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim

        self.fc1 = nn.Linear(state_dim, 400)  # self.hidden_s
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(400, 300)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(300, action_dim)
        self.fc3.weight.data.uniform_(-DDPG_EPS, DDPG_EPS)
        self.tanh = nn.Tanh()

    def forward(self, state):
        """
        returns policy function Pi(s) obtained from actor network
        this function is a gaussian prob distribution for all actions
        with mean lying in (-1,1) and sigma lying in (0,1)
        The sampled action can , then later be rescaled
        :param state: Input state (Torch Variable : [n,state_dim] )
        :return: Output action (Torch Variable: [n,action_dim] )
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        action = self.tanh(x)

        action = action.cpu().data.numpy() * self.action_lim
        action = torch.FloatTensor(action)

        return action
