import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch as t
import numpy as np
import scipy


def threshold_optimization(data, quantization_level=255, n_trial=300, eps=1e-10):
    '''
    This function collect the activation data and find the optimized clipping 
    threshold using KL_div as metric. Since this method is originated from 
    post-training quantization which adopted in Tensor-RT, we keep the number of 
    bits here.
    Args:
        data(numpy array): activation data
        n_bit(int): 
        n_trial(int): the searching steps.
        eps(float): add eps at the average bin step for numberical stability.
    
    '''

    n_lvl = quantization_level+1  # quantization levels
    n_half_lvls = (quantization_level+1)//2
    n_bin_edge = n_lvl * n_trial + 1

    data_max = np.max(np.abs(data))
    hist, bin_edge = np.histogram(data.flatten(),
                                  bins=np.linspace(-data_max,
                                                   data_max,
                                                   num=n_bin_edge))

    mid_idx = int((len(hist)) / 2)
    start_idx = 100
    # log the threshold and corresponding KL-divergence
    kl_result = np.empty([len(range(start_idx, n_trial + 1)), 2])

    for i in range(start_idx, n_trial + 1):
        ref_dist = np.copy(hist[mid_idx - i * n_half_lvls:mid_idx +
                                i * n_half_lvls])
        # merge the outlier
        ref_dist[0] += hist[:mid_idx - i * n_half_lvls].sum()
        ref_dist[-1] += hist[mid_idx + i * n_half_lvls:].sum()
        # perform quantization: bins merge and expansion
        reshape_dim = int(len(ref_dist) / n_lvl)
        ref_dist_reshape = ref_dist.reshape(n_lvl, i)
        # merge bins for quantization
        ref_dist_merged = ref_dist_reshape.sum(axis=1)
        nonzero_mask = (ref_dist_reshape != 0
                        )  # obtain the mask of non-zero bins
        # in each merged large bin, get the average bin_count
        average_bin_count = ref_dist_merged / (nonzero_mask.sum(1) + eps)
        # expand the merged bins back
        expand_bin_count = np.expand_dims(average_bin_count,
                                          axis=1).repeat(i, axis=1)
        candidate_dist = (nonzero_mask * expand_bin_count).flatten()
        kl_div = scipy.stats.entropy(candidate_dist / candidate_dist.sum(),
                                     ref_dist / ref_dist.sum())
        #log threshold and KL-divergence
        current_th = np.abs(
            bin_edge[mid_idx - i * n_half_lvls])  # obtain current threshold
        kl_result[i -
                  start_idx, 0], kl_result[i -
                                           start_idx, 1] = current_th, kl_div

    # based on the logged kl-div result, find the threshold correspond to the smallest kl-div
    th_sel = kl_result[kl_result[:, 1] == kl_result[:, 1].min()][0, 0]
    print("Threshold calibration of current layer finished!")

    return th_sel


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class LsqQuanAct(t.nn.Module):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
        super().__init__()
        self.bit = bit
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.s = t.nn.Parameter(torch.tensor(1.0),requires_grad=True)
        self.isinit = False
        self.symmetric = symmetric
        
    def __repr__(self):
        return f"LsqQuan(thd_pos={self.thd_pos}, thd_neg={self.thd_neg}, s={self.s.data}, per_channel={self.per_channel})"

    
    def init_from(self, x, weight=True, *args, **kwargs):
        # threshold = threshold_optimization(np.array(x.detach().cpu()), quantization_level=int(self.thd_pos), n_trial=300, eps=1e-10)
        # self.s.data = torch.tensor(threshold / (self.thd_pos),dtype=torch.float32).cuda()
        # print("init_from")
        if self.bit >= 16:
            return 
        
        if weight == True:
            if self.per_channel:
                self.s = t.nn.Parameter(
                    x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
            else:
                self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))
        else:
            pass
            # if self.bit > 4:
            # self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))
            # print(self.s)
            threshold = threshold_optimization(np.array(x.detach().cpu()), quantization_level=int(self.thd_pos), n_trial=1000, eps=1e-10)
            self.s.data = torch.tensor(threshold / (self.thd_pos),dtype=torch.float32).cuda()
        #     # print(torch.quantile(x.detach().mean(dim=0),0.95), torch.quantile(x.detach().mean(dim=0),0.05))
        #     self.s = t.nn.Parameter( (torch.quantile(x.detach().mean(dim=0),0.99)-torch.quantile(x.detach().mean(dim=0),0.01)) / ((self.thd_pos - self.thd_neg)))
        #     print(self.s)

    def forward(self, x, clip=True):
        if self.bit >= 16:
            self.s.data = torch.tensor(1.0).to(x.device)
            return x
                
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        
        s_scale = grad_scale(self.s, s_grad_scale)
        # print(s_scale,s_scale.grad)
        # print("self.thd_neg",self.thd_neg, "self.thd_pos", self.thd_pos)
        x = x / s_scale
        if clip:
            x = t.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * s_scale
        return x


class LsqQuan(t.nn.Module):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
        super().__init__()
        self.bit = bit
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.s = t.nn.Parameter(torch.tensor(1.0),requires_grad=True)
        self.isinit = False
        
    def __repr__(self):
        return f"LsqQuan(thd_pos={self.thd_pos}, thd_neg={self.thd_neg}, s={self.s.data}, per_channel={self.per_channel})"

    
    def init_from(self, x, weight=True, *args, **kwargs):

        # threshold = threshold_optimization(np.array(x.detach().cpu()), quantization_level=int(self.thd_pos), n_trial=300, eps=1e-10)
        # self.s.data = torch.tensor(threshold / (self.thd_pos),dtype=torch.float32).cuda()
        if self.per_channel:
            self.s = t.nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
        else:
            self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))
            # if self.bit > 4:
            # self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))
            # print(self.s)
            # threshold = threshold_optimization(np.array(x.detach().cpu()), quantization_level=int(self.thd_pos), n_trial=1000, eps=1e-10)
            # self.s.data = torch.tensor(threshold / (self.thd_pos),dtype=torch.float32).cuda()
        #     # print(torch.quantile(x.detach().mean(dim=0),0.95), torch.quantile(x.detach().mean(dim=0),0.05))
        #     self.s = t.nn.Parameter( (torch.quantile(x.detach().mean(dim=0),0.99)-torch.quantile(x.detach().mean(dim=0),0.01)) / ((self.thd_pos - self.thd_neg)))
        #     print(self.s)

    def forward(self, x, clip=True):
        if self.bit >= 16:
            self.s.data = torch.tensor(1.0).to(x.device)
            return x
                
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        
        s_scale = grad_scale(self.s, s_grad_scale)
        # print(s_scale,s_scale.grad)
        # print("self.thd_neg",self.thd_neg, "self.thd_pos", self.thd_pos)
        x = x / s_scale
        if clip:
            x = t.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * s_scale
        return x
