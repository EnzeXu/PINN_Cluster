import torch
import time
from datetime import datetime
import random
import argparse
import math
import os
import json
import numpy as np
import copy
import pickle
import sys
import matplotlib.pyplot as plt
from torch import nn, optim
from scipy import stats
from tqdm import tqdm
from collections import OrderedDict
from torch.backends import cudnn
from scipy.integrate import odeint


from utils import *


class ConfigSIRAges:
    def __init__(self):
        self.model_name = "ModelSIRAges_Cluster"
        self.T = 100.0
        self.T_all = self.T
        self.T_unit = 0.1
        self.N = int(self.T / self.T_unit)
        self.S_start = 50.0  # 99.0
        self.I_start = 40.0  # 1.0
        self.R_start = 10.0  # 0.0
        self.NN = self.S_start + self.I_start + self.R_start
        self.beta = 0.01  # 0.01
        self.gamma = 0.05  # 0.05
        self.ub = self.T
        self.lb = 0.0
        self.mu = 0.03
        self.lam = self.mu * self.NN

        self.M = np.asarray([
            [19.200, 4.800, 5.050, 3.400, 1.700],
            [4.800, 42.400, 5.900, 6.250, 1.733],
            [5.050, 5.900, 14.000, 7.575, 1.700],
            [3.400, 6.250, 7.575, 9.575, 1.544],
            [1.700, 1.733, 1.700, 1.544, 5.456],
        ])
        self.n = len(self.M)

        self.seed = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gt_real = GroundTruthSIRAges(self.T, self.N)
        self.gt_real_data = torch.tensor(self.gt_real.data).to(self.device)
        self.x_real = torch.tensor(
            np.asarray([[i * self.T_unit] * (self.n * 3) for i in range(self.N)]) / self.T_all * 2.0 - 1.0).float().to(
            self.device)

        self.only_truth_flag = False
        self.truth_rate = 0.001
        self.truth_length = int(self.truth_rate * self.T / self.T_unit)

        self.continuous_flag = False
        self.continue_period = 0.2
        self.round_bit = 3
        self.continue_id = None
        self.mapping_overall_flag = True

        self.sliding_window_flag = False
        self.sw_normal_flag = False
        self.normal_sliding_window_step = 50000
        self.sw_brick_flag = False
        self.epoch_max = None


def RMSELoss(y_pred, y):
    return torch.sqrt(torch.mean((y_pred - y) ** 2) + 1e-12)


class SimpleNetworkSIRAges(nn.Module):
    def __init__(self, config, args=None, truth=None):
        super(SimpleNetworkSIRAges, self).__init__()
        self.config = config
        self.args = args
        myprint("self.truth_length: {} of {} all ".format(self.config.truth_length, self.config.N), self.args.log_path)
        self.setup_seed(self.config.seed)
        self.device = self.config.device
        self.x, self.y0, self.t0 = None, None, None
        self.accurate_x = None
        self.generate_x()
        # self.optimizer = optim.LBFGS(self.parameters(), lr=0.001, max_iter=5000, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)
        self.initial_start()
        self.model_name = self.config.model_name
        self.gt = GroundTruthSIRAges(self.config.T, self.config.N)
        self.gt_data = torch.Tensor(self.gt.data).to(self.device)
        self.y_record = None
        self.sig = nn.Tanh()

        self.loss_norm = RMSELoss
        self.truth = truth if truth else [[], []]
        self.truth_dic = {round(self.truth[0][i], self.config.round_bit): self.truth[1][i] for i in
                          range(len(self.truth[0]))}

        self.network_unit = 20
        # Design A
        # id = 1
        self.fc_x1_id1 = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(1, self.network_unit),
            'sig1': self.sig,
            'lin2': nn.Linear(self.network_unit, self.network_unit),
            'sig2': self.sig,
            'lin3': nn.Linear(self.network_unit, self.network_unit),
            'sig3': self.sig,
            'lin4': nn.Linear(self.network_unit, 1),
        }))

        self.fc_x2_id1 = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(1, self.network_unit),
            'sig1': self.sig,
            'lin2': nn.Linear(self.network_unit, self.network_unit),
            'sig2': self.sig,
            'lin3': nn.Linear(self.network_unit, self.network_unit),
            'sig3': self.sig,
            'lin4': nn.Linear(self.network_unit, 1),
        }))

        self.fc_x3_id1 = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(1, self.network_unit),
            'sig1': self.sig,
            'lin2': nn.Linear(self.network_unit, self.network_unit),
            'sig2': self.sig,
            'lin3': nn.Linear(self.network_unit, self.network_unit),
            'sig3': self.sig,
            'lin4': nn.Linear(self.network_unit, 1),
        }))

        # id = 2
        self.fc_x1_id2 = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(1, self.network_unit),
            'sig1': self.sig,
            'lin2': nn.Linear(self.network_unit, self.network_unit),
            'sig2': self.sig,
            'lin3': nn.Linear(self.network_unit, self.network_unit),
            'sig3': self.sig,
            'lin4': nn.Linear(self.network_unit, 1),
        }))

        self.fc_x2_id2 = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(1, self.network_unit),
            'sig1': self.sig,
            'lin2': nn.Linear(self.network_unit, self.network_unit),
            'sig2': self.sig,
            'lin3': nn.Linear(self.network_unit, self.network_unit),
            'sig3': self.sig,
            'lin4': nn.Linear(self.network_unit, 1),
        }))

        self.fc_x3_id2 = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(1, self.network_unit),
            'sig1': self.sig,
            'lin2': nn.Linear(self.network_unit, self.network_unit),
            'sig2': self.sig,
            'lin3': nn.Linear(self.network_unit, self.network_unit),
            'sig3': self.sig,
            'lin4': nn.Linear(self.network_unit, 1),
        }))

        # id = 3
        self.fc_x1_id3 = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(1, self.network_unit),
            'sig1': self.sig,
            'lin2': nn.Linear(self.network_unit, self.network_unit),
            'sig2': self.sig,
            'lin3': nn.Linear(self.network_unit, self.network_unit),
            'sig3': self.sig,
            'lin4': nn.Linear(self.network_unit, 1),
        }))

        self.fc_x2_id3 = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(1, self.network_unit),
            'sig1': self.sig,
            'lin2': nn.Linear(self.network_unit, self.network_unit),
            'sig2': self.sig,
            'lin3': nn.Linear(self.network_unit, self.network_unit),
            'sig3': self.sig,
            'lin4': nn.Linear(self.network_unit, 1),
        }))

        self.fc_x3_id3 = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(1, self.network_unit),
            'sig1': self.sig,
            'lin2': nn.Linear(self.network_unit, self.network_unit),
            'sig2': self.sig,
            'lin3': nn.Linear(self.network_unit, self.network_unit),
            'sig3': self.sig,
            'lin4': nn.Linear(self.network_unit, 1),
        }))

        # id = 4
        self.fc_x1_id4 = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(1, self.network_unit),
            'sig1': self.sig,
            'lin2': nn.Linear(self.network_unit, self.network_unit),
            'sig2': self.sig,
            'lin3': nn.Linear(self.network_unit, self.network_unit),
            'sig3': self.sig,
            'lin4': nn.Linear(self.network_unit, 1),
        }))

        self.fc_x2_id4 = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(1, self.network_unit),
            'sig1': self.sig,
            'lin2': nn.Linear(self.network_unit, self.network_unit),
            'sig2': self.sig,
            'lin3': nn.Linear(self.network_unit, self.network_unit),
            'sig3': self.sig,
            'lin4': nn.Linear(self.network_unit, 1),
        }))

        self.fc_x3_id4 = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(1, self.network_unit),
            'sig1': self.sig,
            'lin2': nn.Linear(self.network_unit, self.network_unit),
            'sig2': self.sig,
            'lin3': nn.Linear(self.network_unit, self.network_unit),
            'sig3': self.sig,
            'lin4': nn.Linear(self.network_unit, 1),
        }))

        # id = 5
        self.fc_x1_id5 = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(1, self.network_unit),
            'sig1': self.sig,
            'lin2': nn.Linear(self.network_unit, self.network_unit),
            'sig2': self.sig,
            'lin3': nn.Linear(self.network_unit, self.network_unit),
            'sig3': self.sig,
            'lin4': nn.Linear(self.network_unit, 1),
        }))

        self.fc_x2_id5 = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(1, self.network_unit),
            'sig1': self.sig,
            'lin2': nn.Linear(self.network_unit, self.network_unit),
            'sig2': self.sig,
            'lin3': nn.Linear(self.network_unit, self.network_unit),
            'sig3': self.sig,
            'lin4': nn.Linear(self.network_unit, 1),
        }))

        self.fc_x3_id5 = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(1, self.network_unit),
            'sig1': self.sig,
            'lin2': nn.Linear(self.network_unit, self.network_unit),
            'sig2': self.sig,
            'lin3': nn.Linear(self.network_unit, self.network_unit),
            'sig3': self.sig,
            'lin4': nn.Linear(self.network_unit, 1),
        }))

    def forward(self, inputs):
        x1_id1, x1_id2, x1_id3, x1_id4, x1_id5, x2_id1, x2_id2, x2_id3, x2_id4, x2_id5, x3_id1, x3_id2, x3_id3, x3_id4, x3_id5 = torch.chunk(
            inputs, 15, 1)

        # Design A
        x1_new_id1 = self.fc_x1_id1(x1_id1)
        x2_new_id1 = self.fc_x2_id1(x2_id1)
        x3_new_id1 = self.fc_x3_id1(x3_id1)

        x1_new_id2 = self.fc_x1_id2(x1_id2)
        x2_new_id2 = self.fc_x2_id2(x2_id2)
        x3_new_id2 = self.fc_x3_id2(x3_id2)

        x1_new_id3 = self.fc_x1_id3(x1_id3)
        x2_new_id3 = self.fc_x2_id3(x2_id3)
        x3_new_id3 = self.fc_x3_id3(x3_id3)

        x1_new_id4 = self.fc_x1_id4(x1_id4)
        x2_new_id4 = self.fc_x2_id4(x2_id4)
        x3_new_id4 = self.fc_x3_id4(x3_id4)

        x1_new_id5 = self.fc_x1_id5(x1_id5)
        x2_new_id5 = self.fc_x2_id5(x2_id5)
        x3_new_id5 = self.fc_x3_id5(x3_id5)

        outputs = torch.cat((x1_new_id1, x1_new_id2, x1_new_id3, x1_new_id4, x1_new_id5, x2_new_id1, x2_new_id2,
                             x2_new_id3, x2_new_id4, x2_new_id5, x3_new_id1, x3_new_id2, x3_new_id3, x3_new_id4,
                             x3_new_id5), 1)
        return outputs

    def generate_x(self):
        x = [[i * self.config.T_unit] * (self.config.n * 3) for i in range(self.config.N)]  # toy
        x = np.asarray(x)
        x = self.encode_t(x)
        self.x = torch.Tensor(x).float().to(self.device)
        self.accurate_x = [i * self.config.T_unit for i in range(self.config.N)]
        myprint("[continuous] self.x: shape = {}".format(self.x.shape), self.args.log_path)

    def initial_start(self):
        self.t0 = torch.Tensor(np.asarray([-1.0, -1.0, -1.0] * self.config.n).reshape([1, -1])).float().to(self.device)
        self.y0 = torch.Tensor(np.asarray(
            [self.config.S_start] * self.config.n + [self.config.I_start] * self.config.n + [
                self.config.R_start] * self.config.n
        ).reshape([1, -1])).float().to(self.device)
        # self.yend = torch.Tensor(np.asarray([0, 0, 100]).reshape([1, -1])).float().to(self.device)
        # self.tend = torch.Tensor(np.asarray([1.0, 1.0, 1.0]).reshape([1, -1])).float().to(self.device)

    def loss_only_ground_truth(self):
        self.eval()
        y = self.forward(self.x)
        self.y_record = y
        # self.loss_norm = torch.nn.MSELoss().to(self.device)
        loss = self.loss_norm(y, self.gt_data)
        self.train()
        return loss, [loss]

    def real_loss(self):
        self.eval()
        y = self.forward(self.config.x_real)
        real_loss = self.loss_norm(y[:self.config.N, :], self.config.gt_real_data[:self.config.N, :])
        return real_loss

    def loss(self, epoch=None):
        self.eval()
        # cp = CheckPoint()
        y = self.forward(self.x)
        self.y_record = y
        # s = y[:, 0:1]
        # i = y[:, 1:2]
        # r = y[:, 2:3]
        # cp.time("c1")
        s = y[:, 0: self.config.n]
        i = y[:, self.config.n: self.config.n * 2]
        r = y[:, self.config.n * 2: self.config.n * 3]

        s_t_collection, i_t_collection, r_t_collection = [], [], []
        for ii in range(self.config.n):
            s_t_collection.append(torch.gradient(s[:, ii:ii + 1].reshape([self.config.N]),
                                                 spacing=(self.decode_t(self.x)[:, 0:1].reshape([self.config.N]),))[
                                      0].reshape([self.config.N, 1]))
            i_t_collection.append(torch.gradient(i[:, ii:ii + 1].reshape([self.config.N]),
                                                 spacing=(self.decode_t(self.x)[:, 0:1].reshape([self.config.N]),))[
                                      0].reshape([self.config.N, 1]))
            r_t_collection.append(torch.gradient(r[:, ii:ii + 1].reshape([self.config.N]),
                                                 spacing=(self.decode_t(self.x)[:, 0:1].reshape([self.config.N]),))[
                                      0].reshape([self.config.N, 1]))
        s_t = torch.cat(s_t_collection, 1)
        i_t = torch.cat(i_t_collection, 1)
        r_t = torch.cat(r_t_collection, 1)

        tmp_s_t_target_collection, tmp_i_t_target_collection, tmp_r_t_target_collection = [], [], []
        for ii in range(self.config.n):
            tmp_s_t_target = torch.zeros([self.config.N, 1]).to(self.device)
            tmp_i_t_target = torch.zeros([self.config.N, 1]).to(self.device)
            tmp_r_t_target = torch.zeros([self.config.N, 1]).to(self.device)
            for jj in range(self.config.n):
                tmp_s_t_target += (-self.config.beta * s[:, ii:ii + 1] * self.config.M[ii][jj] * i[:,
                                                                                                 jj:jj + 1]) / self.config.NN
                tmp_i_t_target += (self.config.beta * s[:, ii:ii + 1] * self.config.M[ii][jj] * i[:,
                                                                                                jj:jj + 1]) / self.config.NN
            tmp_i_t_target -= self.config.gamma * i[:, ii:ii + 1]
            tmp_r_t_target += self.config.gamma * i[:, ii:ii + 1]
            tmp_s_t_target += (- self.config.mu * s[:, ii:ii + 1] + self.config.lam)
            tmp_i_t_target += (- self.config.mu * i[:, ii:ii + 1])
            tmp_r_t_target += (- self.config.mu * r[:, ii:ii + 1])
            tmp_s_t_target_collection.append(tmp_s_t_target)
            tmp_i_t_target_collection.append(tmp_i_t_target)
            tmp_r_t_target_collection.append(tmp_r_t_target)
        s_t_target = torch.cat(tmp_s_t_target_collection, 1)
        i_t_target = torch.cat(tmp_i_t_target_collection, 1)
        r_t_target = torch.cat(tmp_r_t_target_collection, 1)

        f_s = s_t - s_t_target
        f_i = i_t - i_t_target
        f_r = r_t - r_t_target
        # cp.time("c3")

        f_y = torch.cat((f_s, f_i, f_r), 1)
        y0_pred = self.forward(self.t0)


        zeros_15D = torch.Tensor([[0.0] * 15] * self.config.N).to(self.device)
        loss_1 = self.loss_norm(y[:self.config.truth_length, :], self.gt_data[:self.config.truth_length, :])
        # cp.time("c4")
        if self.config.sliding_window_flag:
            if self.config.sw_normal_flag:
                self.loss_2_weight_numpy = generate_normal_distribution_weight(self.config.N, self.config.n * 3, (
                            epoch % int(
                        self.config.normal_sliding_window_step)) / self.config.normal_sliding_window_step)
                loss_2 = self.loss_norm(f_y * torch.Tensor(self.loss_2_weight_numpy).to(self.device), zeros_15D)
            else:
                self.loss_2_weight_numpy = generate_brick_distribution_weight(self.config.N, self.config.n * 3,
                                                                              epoch / self.config.epoch_max)
                loss_2 = self.loss_norm(f_y * torch.Tensor(self.loss_2_weight_numpy).to(self.device), zeros_15D)

        else:
            loss_2 = self.loss_norm(f_y, zeros_15D)  # torch.mean(torch.square(f_y))  # + torch.var(torch.square(f_y))

        loss_3 = self.loss_norm(torch.abs(y),
                                y)  # torch.mean(torch.square((torch.abs(s) - s))) + torch.mean(torch.square((torch.abs(i) - i))) + torch.mean(torch.square((torch.abs(r) - r))) #+ torch.mean(torch.square((0.1/(s * s)))) + torch.mean(torch.square((0.1/(i * i)))) + torch.mean(torch.square((0.1/(r * r))))
        # loss_4 = torch.mean(torch.square(0.00001 / ((s * s + i * i) * (i * i + r * r) * (s * s + r * r) + 1e-8)))
        loss_4 = self.match_truth(self.accurate_x, y.cpu().detach().numpy())
        # cp.time("c5")
        # 04/27 TODO: re-design the loss functions. I want to do it but Chen asked me to implement new models. Now it's your turn.

        loss = loss_1 + loss_2 + loss_3 + loss_4
        self.train()
        return loss, [loss_1, loss_2, loss_3, loss_4]
        # return torch.mean(torch.square(y_hat - y))
        # return F.mse_loss(torch.cat((u_hat, v_hat), 1), torch.cat((u, v), 1))
        # return torch.abs(u_hat - u) + torch.abs(v_hat - v)  # F.mse_loss(x_hat, x) + beta * self.kl_div(rho)

    def match_truth(self, x, y):
        if len(self.truth[0]) == 0:
            return torch.Tensor([0.0]).to(self.device)
        diff = [np.abs(y_tmp - self.truth_dic.get(round(x_tmp, self.config.round_bit))) for x_tmp, y_tmp in zip(x, y) if
                round(x_tmp, self.config.round_bit) in self.truth_dic]
        diff = torch.Tensor(diff).to(self.device)
        zeros_2D = torch.Tensor([[0.0] * self.config.n * 3] * len(diff)).to(self.device)
        loss_truth_match = self.loss_norm(diff, zeros_2D)
        if len(diff) != len(self.truth[0]):
            myprint("Error: match_truth: {} / {} items to match, loss_truth_match = {}".format(
                len(diff),
                len(self.truth[0]),
                loss_truth_match.item()), self.args.log_path)
        return loss_truth_match

    def encode_t(self, num):
        if not self.config.mapping_overall_flag:
            return (num - self.config.lb) / (self.config.ub - self.config.lb) * 2.0 - 1.0
        return num / self.config.T_all * 2.0 - 1.0

    def decode_t(self, num):
        if not self.config.mapping_overall_flag:
            return self.config.lb + (num + 1.0) / 2.0 * (self.config.ub - self.config.lb)
        return (num + 1.0) / 2.0 * self.config.T_all

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.deterministic = True


class GroundTruthSIRAges:
    def __init__(self, t_max, length):
        S_start = 50.0  # 99.0
        I_start = 40.0  # 1.0
        R_start = 10  # 1.0 # 0.0
        NN = S_start + I_start + R_start
        beta = 0.01
        gamma = 0.05
        mu = 0.03
        lam = mu * NN

        M = np.asarray([
            [19.200, 4.800, 5.050, 3.400, 1.700],
            [4.800, 42.400, 5.900, 6.250, 1.733],
            [5.050, 5.900, 14.000, 7.575, 1.700],
            [3.400, 6.250, 7.575, 9.575, 1.544],
            [1.700, 1.733, 1.700, 1.544, 5.456],
        ])
        self.n = len(M)
        y0 = np.asarray([S_start] * self.n + [I_start] * self.n + [R_start] * self.n)
        self.t = np.linspace(0, t_max, length)
        self.data = odeint(self.pend, y0, self.t, args=(M, beta, gamma, NN, self.n, mu, lam))

    @staticmethod
    def pend(y, t, M, beta, gamma, NN, n, mu, lam):
        map = y
        # dydt = np.asarray([(a * map[0]) - c * map[0] * map[1], - b * map[1] + d * c * map[0] * map[1]])
        S_arr = y[0: n]
        I_arr = y[n: 2 * n]
        R_arr = y[2 * n: 3 * n]
        ds = []
        di = []
        dr = []
        for i in range(n):
            ds.append(- beta * S_arr[i] / NN * sum([M[i][j] * I_arr[j] for j in range(n)]) - mu * S_arr[i] + lam)
            di.append(
                beta * S_arr[i] / NN * sum([M[i][j] * I_arr[j] for j in range(n)]) - gamma * I_arr[i] - mu * I_arr[i])
            dr.append(gamma * I_arr[i] - mu * R_arr[i])
        dydt = np.asarray(ds + di + dr)
        return dydt

    def print_truth(self):
        y_lists = [self.data[:, i] for i in range(3 * self.n)]
        x_list = self.t
        color_list = ["red"] * self.n + ["blue"] * self.n + ["green"] * self.n
        labels = ["0-9", "10-19", "20-39", "40-59", "60+"]
        legend_list = ["S{}({})".format(i + 1, labels[i]) for i in range(self.n)] + ["I{}({})".format(i + 1, labels[i])
                                                                                     for i in range(self.n)] + [
                          "R{}({})".format(i + 1, labels[i]) for i in range(self.n)]
        line_style_list = ["dashed", "dotted", "dashdot", (0, (3, 1, 1, 1, 1, 1)), (0, (3, 10, 1, 10))] * 3

        draw_two_dimension(
            y_lists=y_lists,
            x_list=x_list,
            color_list=color_list,
            legend_list=legend_list,
            line_style_list=line_style_list,
            fig_title="Ground Truth: SIR - Ages",
            fig_size=(8, 6),
            show_flag=False,
            save_flag=False,
            save_path=None
        )


def get_now_string():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))


def generate_normal_distribution_weight(length, width, mu_location):
    mu = mu_location * length
    sigma = length // 10
    x = np.linspace(0, length, length)
    y = 1e2 * length * stats.norm.pdf(x, mu, sigma)
    y = y / y.sum() * length
    res = np.column_stack([y for i in range(width)])
    return res


def generate_brick_distribution_weight(length, width, epoch_ratio, epoch_freeze=0.15, epoch_moving=0.05,
                                       epoch_final=0.40):
    # epoch_ratio = epoch / epoch_max
    if epoch_ratio <= epoch_freeze:
        weight_range = [0, 0.25 * length]
    elif epoch_freeze < epoch_ratio <= epoch_freeze + epoch_moving:
        weight_range = [0, (0.25 + (epoch_ratio - (1 * epoch_freeze)) / epoch_moving * 0.25) * length]
    elif epoch_freeze + epoch_moving < epoch_ratio <= 2 * epoch_freeze + epoch_moving:
        weight_range = [0, 0.5 * length]
    elif 2 * epoch_freeze + epoch_moving < epoch_ratio <= 2 * epoch_freeze + 2 * epoch_moving:
        weight_range = [0, (0.5 + (epoch_ratio - (2 * epoch_freeze + epoch_moving)) / epoch_moving * 0.25) * length]
    elif 2 * epoch_freeze + 2 * epoch_moving < epoch_ratio <= 3 * epoch_freeze + 2 * epoch_moving:
        weight_range = [0, 0.75 * length]
    elif 3 * epoch_freeze + 2 * epoch_moving < epoch_ratio <= 3 * epoch_freeze + 3 * epoch_moving:
        weight_range = [0,
                        (0.75 + (epoch_ratio - (3 * epoch_freeze + 2 * epoch_moving)) / epoch_moving * 0.25) * length]
    elif 3 * epoch_freeze + 3 * epoch_moving < epoch_ratio <= 1.0:
        weight_range = [0, 1.0 * length]

    weight_high = length * 1 / (weight_range[1] - weight_range[0])
    y = [weight_high if i < weight_range[1] else 0 for i in range(length)]
    y = np.asarray(y)
    y = y / y.sum() * length
    res = np.column_stack([y for i in range(width)])
    return res


def train_sir_ages(model, args, config, now_string, resume_epoch=None, resume_loss_record=None,
                   resume_real_loss_record=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model_framework(config).to(device)
    model.train()
    model_save_path_last = f"{args.main_path}/train/{model.model_name}_{args.epoch}_{args.epoch_step}_{args.lr}_{config.beta}_{config.gamma}_{now_string}_last.pt"
    model_save_path_best = f"{args.main_path}/train/{model.model_name}_{args.epoch}_{args.epoch_step}_{args.lr}_{config.beta}_{config.gamma}_{now_string}_best.pt"
    loss_save_path = f"{args.main_path}/loss/{model.model_name}_{args.epoch}_{args.epoch_step}_{args.lr}_{now_string}_loss_{args.epoch}.npy"
    real_loss_save_path = f"{args.main_path}/loss/{model.model_name}_{args.epoch}_{args.epoch_step}_{args.lr}_{now_string}_real_loss_{args.epoch}.npy"
    y_record_save_path = f"{args.main_path}/loss/{model.model_name}_{args.epoch}_{args.epoch_step}_{args.lr}_{now_string}_y_record.npy"
    myprint("using " + str(device), args.log_path)
    myprint("epoch = {}".format(args.epoch), args.log_path)
    myprint("epoch_step = {}".format(args.epoch_step), args.log_path)
    myprint("model_name = {}".format(model.model_name), args.log_path)
    myprint("now_string = {}".format(now_string), args.log_path)
    myprint("model_save_path_last = {}".format(model_save_path_last), args.log_path)
    myprint("model_save_path_best = {}".format(model_save_path_best), args.log_path)
    myprint("loss_save_path = {}".format(loss_save_path), args.log_path)
    myprint("real_loss_save_path = {}".format(real_loss_save_path), args.log_path)
    myprint("y_record_save_path = {}".format(y_record_save_path), args.log_path)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    initial_lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch / 10000 + 1))
    # scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01, step_size=1000)
    # optimizer = optim.LBFGS(model.parameters(), lr=args.lr, max_iter=5000, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100,
    #       line_search_fn=None)
    epoch_step = args.epoch_step
    start_time = time.time()
    start_time_0 = start_time
    best_loss = np.inf
    loss_record = []
    real_loss_record = []
    start_index = 1
    if resume_epoch:
        for i in range(resume_epoch):
            scheduler.step()
        loss_record = resume_loss_record
        real_loss_record = resume_real_loss_record
        start_index = resume_epoch + 1
        myprint("[Resume] loss_record length: {}".format(len(loss_record)), args.log_path)
        myprint("[Resume] real_loss_record length: {}".format(len(real_loss_record)), args.log_path)
        myprint("[Resume] start_index: {}".format(start_index), args.log_path)
    # y_record = []
    for epoch in range(start_index, args.epoch + 1):
        # cp = CheckPoint()
        optimizer.zero_grad()
        # cp.time("a")
        inputs = model.x
        outputs = model(inputs)
        # cp.time("b")
        # u_hat, v_hat = torch.chunk(outputs, 2, 1)
        if config.only_truth_flag:
            loss, loss_list = model.loss_only_ground_truth()
        else:
            loss, loss_list = model.loss(epoch)
        # cp.time("c")
        real_loss = model.real_loss()
        # cp.time("d")
        loss.backward()
        # cp.time("e")
        optimizer.step()
        # cp.time("f")
        scheduler.step()
        # cp.time("g")
        loss_record.append(float(loss.item()))
        real_loss_record.append(float(real_loss.item()))
        # y_record.append(model.y_record.cpu().detach().numpy())
        # cp.time("h")
        if epoch % epoch_step == 0:
            now_time = time.time()
            loss_print_part = " ".join(
                ["Loss_{0:d}:{1:.6f}".format(i + 1, loss_part.item()) for i, loss_part in enumerate(loss_list)])
            myprint(
                "Epoch [{0:05d}/{1:05d}] Loss:{2:.6f} {3} Lr:{4:.6f} Time:{5:.6f}s ({6:.2f}min in total, {7:.2f}min remains)".format(
                    epoch, args.epoch, loss.item(), loss_print_part, optimizer.param_groups[0]["lr"],
                    now_time - start_time, (now_time - start_time_0) / 60.0,
                    (now_time - start_time_0) / 60.0 / (epoch - start_index + 1) * (args.epoch - epoch)), args.log_path)
            start_time = time.time()
            torch.save(
                {
                    'epoch': args.epoch,
                    'model_state_dict': model.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item()
                }, model_save_path_last)
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(
                    {
                        'epoch': args.epoch,
                        'model_state_dict': model.state_dict(),
                        # 'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item()
                    }, model_save_path_best)
        if epoch % args.save_step == 0 or epoch == args.epoch:
            test_sir_ages(model, args, config, now_string, True, model.gt)
            myprint("[Loss]", args.log_path)
            draw_loss(np.asarray(loss_record))
            myprint("[Real Loss]", args.log_path)
            draw_loss(np.asarray(real_loss_record))
            np.save(loss_save_path, np.asarray(loss_record))
            np.save(real_loss_save_path, np.asarray(real_loss_record))
            # np.save(y_record_save_path, np.asarray(y_record))
    loss_record = np.asarray(loss_record)
    real_loss_record = np.asarray(real_loss_record)
    res_dic = {
        "start_time": start_time_0,
        "epoch": args.epoch,
        "model_save_path_last": model_save_path_last,
        "model_save_path_best": model_save_path_best,
        "loss_save_path": loss_save_path,
        "real_loss_save_path": real_loss_save_path,
        "best_loss": best_loss,
        "loss_record": loss_record,
        "real_loss_record": real_loss_record
    }
    return model, res_dic


def test_sir_ages(model, args, config, now_string, show_flag=True, gt=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model_framework(config).to(device)
    # model_save_path = f"{args.main_path}/train/{model.model_name}_{args.epoch}_{args.epoch_step}_{args.lr}_{config.beta}_{config.gamma}_{now_string}_last.pt"
    # model.load_state_dict(torch.load(model_save_path, map_location=device)["model_state_dict"])
    model.eval()
    myprint("Testing & drawing...", args.log_path)
    t = model.x
    y = model(t)
    y0_pred = model(model.t0)
    s, i, r = y[:, 0:5], y[:, 5:10], y[:, 10:15]
    s = s.cpu().detach().numpy()
    i = i.cpu().detach().numpy()
    r = r.cpu().detach().numpy()
    x = model.decode_t(t).cpu().detach().numpy()
    s_pred = [s[:, id:id + 1].reshape([model.config.N]) for id in range(model.config.n)]
    i_pred = [i[:, id:id + 1].reshape([model.config.N]) for id in range(model.config.n)]
    r_pred = [r[:, id:id + 1].reshape([model.config.N]) for id in range(model.config.n)]
    x = x[:, 0:1].reshape([model.config.N])

    figure_save_path = f"{args.main_path}/figure/{model.model_name}_{args.epoch}_{args.epoch_step}_{args.lr}_{config.beta}_{config.gamma}_{now_string}_{int(time.time())}.png"
    labels = ["0-9", "10-19", "20-39", "40-59", "60+"]
    color_list = ["grey"] * (3 * model.config.n) + ["red"] * model.config.n + ["blue"] * model.config.n + [
        "green"] * model.config.n
    legend_list = ["S{}({})_truth".format(i + 1, labels[i]) for i in range(model.config.n)] + [
        "I{}({})_truth".format(i + 1, labels[i]) for i in range(model.config.n)] + [
                      "R{}({})_truth".format(i + 1, labels[i]) for i in range(model.config.n)] + \
                  ["S{}({})".format(i + 1, labels[i]) for i in range(model.config.n)] + [
                      "I{}({})".format(i + 1, labels[i]) for i in range(model.config.n)] + [
                      "R{}({})".format(i + 1, labels[i]) for i in range(model.config.n)]
    line_style_list = ["dashed"] * (3 * model.config.n) + ["dashed", "dotted", "dashdot", (0, (3, 1, 1, 1, 1, 1)),
                                                           (0, (3, 10, 1, 10))] * 3
    y_truth = [gt.data[:, id:id + 1].reshape([model.config.N]) for id in range(model.config.n * 3)]
    if not model.config.sliding_window_flag:
        draw_two_dimension(
            y_lists=y_truth + s_pred + i_pred + r_pred,
            x_list=x,
            color_list=color_list,
            legend_list=legend_list,
            line_style_list=line_style_list,
            fig_title="Predict: SIR - Ages",
            fig_size=(24, 18),
            show_flag=False,
            save_flag=True,
            save_path=figure_save_path
        )
    else:
        sw_weight = [item[0] for item in model.loss_2_weight_numpy]
        draw_two_dimension(
            y_lists=[sw_weight] + y_truth + s_pred + i_pred + r_pred,
            x_list=x,
            color_list=["black"] + color_list,
            legend_list=["sliding window weights"] + legend_list,
            line_style_list=["dotted"] + line_style_list,
            fig_title="Predict: SIR - Ages",
            fig_size=(24, 18),
            show_flag=False,
            save_flag=True,
            save_path=figure_save_path
        )



# class Args:
#     def __init__(self):
#         self.epoch = 2000000  # 500000 # 500
#         self.epoch_step = 5000  # 1
#         self.lr = 0.01
#         self.main_path = "."
#         self.save_step = 50000  # 10000


def draw_loss(loss_list):
    draw_two_dimension(
        y_lists=[loss_list],
        x_list=range(1, len(loss_list) + 1),
        color_list=["blue"],
        legend_list=["loss"],
        line_style_list=["solid"],
        fig_title="Loss - {} epochs".format(len(loss_list)),
        fig_x_label="epoch",
        fig_y_label="loss",
        fig_size=(8, 6),
        show_flag=False,
        save_flag=False,
        save_path=None
    )


def run_sir_ages(args=None):
    # args = Args()
    if not os.path.exists("{}/train".format(args.main_path)):
        os.makedirs("{}/train".format(args.main_path))
    if not os.path.exists("{}/figure".format(args.main_path)):
        os.makedirs("{}/figure".format(args.main_path))
    if not os.path.exists("{}/loss".format(args.main_path)):
        os.makedirs("{}/loss".format(args.main_path))
    now_string = get_now_string()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ConfigSIRAges()
    model = SimpleNetworkSIRAges(config, args).to(device)
    model, res_dic = train_sir_ages(model, args, config, now_string)
    return res_dic


# def run_sir_ages_sliding_window(main_path=None, sw_type="normal", args=None):
#     # args = Args()
#     if main_path:
#         args.main_path = main_path
#     if not os.path.exists("{}/train".format(args.main_path)):
#         os.makedirs("{}/train".format(args.main_path))
#     if not os.path.exists("{}/figure".format(args.main_path)):
#         os.makedirs("{}/figure".format(args.main_path))
#     now_string = get_now_string()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     config = ConfigSIRAges()
#     config.sliding_window_flag = True
#     config.epoch_max = args.epoch
#     if sw_type == "normal":
#         config.sw_normal_flag = True
#     elif sw_type == "brick":
#         config.sw_brick_flag = True
#     model = SimpleNetworkSIRAges(config).to(device)
#     model, res_dic = train_sir_ages(model, args, config, now_string)
#     return res_dic
#     # model = SimpleNetworkSIRAges(config).to(device)
#     # test_sir_ages(model, args, config, now_string)


def run_sir_ages_continue(args=None):
    args_0 = copy.deepcopy(args)
    if not os.path.exists("{}/train".format(args_0.main_path)):
        os.makedirs("{}/train".format(args_0.main_path))
    if not os.path.exists("{}/figure".format(args_0.main_path)):
        os.makedirs("{}/figure".format(args_0.main_path))
    if not os.path.exists("{}/loss".format(args_0.main_path)):
        os.makedirs("{}/loss".format(args_0.main_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    now_string_list = []
    now_string = None
    config_0 = ConfigSIRAges()
    continue_n = int(1.0 / config_0.continue_period)
    truth_x = []
    truth_y = []

    real_loss_record_list = []
    for i in range(continue_n):

        config = ConfigSIRAges()
        config.T = config_0.T * config_0.continue_period * (i + 1)
        config.N = int(config.T / config.T_unit)
        config.ub = config.T
        config.continue_id = i
        args = copy.deepcopy(args_0)
        args.epoch = int(args.epoch * config_0.continue_period)

        now_string = get_now_string()

        myprint("i = {}, length of truth = {} now".format(i, len(truth_x)), args.log_path)

        model = SimpleNetworkSIRAges(config, args, [truth_x, truth_y]).to(device)
        if i > 0:
            model_state_dict_path = f"{args.main_path}/train/{config.model_name}_{args.epoch}_{args.epoch_step}_{args.lr}_{config.beta}_{config.gamma}_{now_string_list[-1]}_best.pt"
            model.load_state_dict(torch.load(model_state_dict_path, map_location=device)["model_state_dict"])
            myprint("Loaded previous trained model from {} successfully!".format(model_state_dict_path), args.log_path)
            myprint("Test before training...", args.log_path)
            myprint("Now string list: {}".format(now_string_list), args.log_path)
            test_sir_ages(model, args, config, now_string_list[-1], True, model.gt)
        now_string_list.append(now_string)
        model, res_dic = train_sir_ages(model, args, config, now_string)
        with open(f"{args.main_path}/train/{config.model_name}_{now_string}_i={i}.model", "wb") as f:
            pickle.dump(model, f)
        real_loss_record_list.append(res_dic["real_loss_record"])
        draw_loss(np.concatenate(real_loss_record_list))
        np.save(f"{args.main_path}/train/{config.model_name}_{now_string}_real_loss_record_i={i}.pt",
                np.concatenate(real_loss_record_list))
        y = model(model.x)
        y = y.cpu().detach().numpy()
        for one_x, one_y in zip(model.accurate_x, y):
            one_x = round(one_x, model.config.round_bit)
            if not one_x in truth_x:
                truth_x.append(one_x)
                truth_y.append(one_y)
    real_loss_record_all = np.concatenate(real_loss_record_list)
    draw_loss(real_loss_record_all)
    real_loss_all_path = f"{args_0.main_path}/train/{config_0.model_name}_{now_string}_real_loss_all.npy"
    np.save(real_loss_all_path, real_loss_record_all)
    myprint("real_loss_all is saved to {} (length={})".format(real_loss_all_path, len(real_loss_record_all)), args.log_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100, help="epoch")
    parser.add_argument("--log_path", type=str, default="logs/1.txt", help="log path")
    parser.add_argument("--mode", type=str, default="origin", help="continue or origin")
    parser.add_argument("--epoch_step", type=int, default=10, help="epoch_step")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
    parser.add_argument("--main_path", default=".", help="main_path")
    parser.add_argument("--save_step", type=int, default=100, help="save_step")
    opt = parser.parse_args()

    myprint("log_path: {}".format(opt.log_path), opt.log_path)
    myprint("cuda is available: {}".format(torch.cuda.is_available()), opt.log_path)

    if opt.mode == "origin":
        run_sir_ages(opt)
    else:
        run_sir_ages_continue(opt)


