import torch
import torch.nn as nn
import torch.fft
import os
import numpy as np
from sklearn.preprocessing import StandardScaler


class Projector_Multi(nn.Module):
    def __init__(self, enc_in, seq_len, hidden_dims = [16, 16], hidden_layers = 2, kernel_size=3):
        super(Projector_Multi, self).__init__()
        
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=False)

        layers = [nn.Linear(2*enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers-1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]
        
        layers += [nn.Linear(hidden_dims[-1], enc_in, bias=False), nn.Tanh()]
        # layers += [nn.Linear(hidden_dims[-1], enc_in, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, period):
        batch_size = x.shape[0]
        x = self.series_conv(x)
        x = torch.cat([x, period.unsqueeze(1)], dim=1)
        x = x.view(batch_size, -1)
        y = self.backbone(x)
        return y


class Projector_Ratio(nn.Module):
    def __init__(self, enc_in, seq_len, hidden_dims = [16, 16], hidden_layers = 2, kernel_size=3):
        super(Projector_Ratio, self).__init__()
        
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv_x = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=False)
        self.series_conv_trend = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=False)
        self.series_conv_period = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=False)

        layers = [nn.Linear(3 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers-1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]
        
        # layers += [nn.Linear(hidden_dims[-1], enc_in, bias=False), nn.Tanh()]
        layers += [nn.Linear(hidden_dims[-1], enc_in, bias=False), nn.Sigmoid()]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, trend, period):
        batch_size = x.shape[0]
        x = self.series_conv_x(x)
        trend = self.series_conv_trend(trend)
        period = self.series_conv_period(period)
        x = torch.cat([x, trend, period], dim=1)
        

        x = x.view(batch_size, -1)
        y = self.backbone(x)
        return y



class RevIN_weight(nn.Module):
    def __init__(self, d_feature, seq_len, k,r):
        super(RevIN_weight, self).__init__()

        self.num_features = d_feature
        self.seq_len = seq_len
        self.eps = 1e-5
        self.weight_rev = True
        self._init_params()
        print("\nRevIN active! weight_rev=", self.weight_rev)
        self.cnt = 0
        self.k = k
        self.r = r

        self.ratioSum = 0
        self.cntSum = 0
    

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features).unsqueeze(0).unsqueeze(0))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features).unsqueeze(0).unsqueeze(0))
        self.period_multi_learner = Projector_Multi(enc_in=self.num_features, seq_len=self.seq_len)
        self.trend_multi_learner = Projector_Multi(enc_in=self.num_features, seq_len=self.seq_len)
        self.trend_period_ratio_learner = Projector_Ratio(enc_in=self.num_features, seq_len=self.seq_len)

    def forward(self, x, mode:str, mask, x_interpolate):
        if mode == 'norm':
            x = self._normalize(x, mask, x_interpolate)
        
        elif mode == 'denorm':
            x = self._denormalize(x)
        
        else: raise NotImplementedError

        return x    


    def SelectRange(self, x, minn, maxx, pdim):
        # x是一个2维tensor，去除x第二维的第pdim索引数字超过minn和maxx的第一维元素
        ind = (((x[:, pdim] < minn) | (x[:, pdim] >= maxx))).nonzero()
        ind = torch.unique(ind)
        ind_r = torch.arange(x.shape[0]).to(x.device)
        # ind_r = torch.arange(x.shape[0])
        ind_r = ind_r[~torch.isin(ind_r, ind)]
        result_tensor = torch.index_select(x, dim=0, index=ind_r)
        return result_tensor


    def NormalDistribution(self, i, multi, sigma):
        # 返回 N(0,sigma)正态分布在py处的值
        return multi * np.exp(-1 * i*i/(2*sigma*sigma)) / (np.sqrt(2*np.pi) * sigma)
    

    def get_weight_general(self, x, mask, period, multi):
        # 根据周期获取缺失值权重
        sigma = 1
        weight = torch.ones_like(x).to(x.device)
        pos = (mask == 0).nonzero()
        # 根据周期添加权重
        for i in range(-self.r, self.r+1):
            if i==0: continue
            pos_py = pos.clone()
            pos_py[:, 1] += i * period[pos_py[:, 0], pos_py[:, 2]].long()
            pos_py = self.SelectRange(pos_py, 0, x.shape[1], 1)
            weight[pos_py[:, 0], pos_py[:, 1], pos_py[:, 2]] += self.NormalDistribution(i, multi[pos_py[:, 0], pos_py[:, 2]], sigma)

        return weight
    
    
    def _get_weight(self, x, mask, x_interpolate):
        # 计算趋势权重
        x_d = x.clone().detach()
        x_d = torch.nn.functional.normalize(x_d, dim=1)
        ones = torch.ones(x.shape[0], x.shape[2], dtype=torch.int64).to(x.device)
        multi_trend = self.trend_multi_learner(x_d, ones).exp()
        weight_trend = self.get_weight_general(x, mask, ones, multi_trend)

        # 计算周期权重
        seq_len = x.shape[1]
        N_2 = seq_len // 2
        # 得到最大幅值对应的周期
        x_fft = torch.fft.fft(x_interpolate, dim=1).to(x.device)
        x_fft = torch.abs(x_fft)[:, :N_2, :]
        frequency = torch.fft.fftfreq(seq_len)[:N_2].to(x.device)

        k=self.k     # 前k大的幅值参与计算
        amplitudes, index = torch.topk(x_fft, k, dim=1)
        index = index.to(x.device)
        amplitudeSum = torch.sum(amplitudes, dim=1) + self.eps
        x_int = x_interpolate.clone().detach()
        x_int = torch.nn.functional.normalize(x_int, dim=1)
        weight_period = torch.zeros(x.shape).to(x.device)

        for i in range(k):
            max_period = torch.reciprocal(frequency[index[:,i,:]]).to(x.device) # B, N
            max_period[max_period >= self.seq_len] = 1
            max_period = max_period.round()
            # print(max_period)
            rate = (amplitudes[:,i,:] / amplitudeSum).unsqueeze(1)

            multi_period = self.period_multi_learner(x_int, max_period).exp()
            sub_weight_period = self.get_weight_general(x, mask, max_period, multi_period)
            weight_period += sub_weight_period*rate

        ratio = self.trend_period_ratio_learner(x_int, weight_trend, weight_period).unsqueeze(1)
        weight = weight_trend * ratio + weight_period * (1-ratio)
        # weight = weight_trend
        self.ratioSum += ratio.sum().item() / (ratio.shape[0] * ratio.shape[1] * ratio.shape[2])
        self.cntSum += 1
        self.cnt = self.cnt + 1
        # if(self.cnt >= 200):
        #     # print("multi_trend", multi_trend[0])
        #     # print("multi_period", multi_period[0])
        #     print("ratio", ratio[0])
        #     self.cnt = 0

        return weight * mask


    def _normalize(self, x, mask, x_interpolate):
        if self.weight_rev:
            weight = self._get_weight(x, mask, x_interpolate)
        else:
            weight = mask

        x_d = x * weight
        # x_d = x * mask
        cnt = torch.sum(weight, dim=1)
        # cnt = torch.sum(mask, dim=1)
        cnt[cnt == 0] = 1

        mean = torch.sum(x_d, dim=1) / cnt
        mean = mean.unsqueeze(1)
        self.mean = mean
        
        x = x - self.mean
        x = x.masked_fill(mask == 0, 0)
        x_d = x * weight
        # x_d = x * mask

        stdev = torch.sqrt(torch.sum(x_d * x_d, dim=1) / cnt + self.eps)
        stdev = stdev.unsqueeze(1)
        self.stdev = stdev
        x = x / self.stdev

        # 调整参数
        # x = x * self.affine_weight
        # x = x + self.affine_bias
        return x, weight * mask

    def _denormalize(self, x):
        # 调整参数
        # x = x - self.affine_bias
        # x = x / (self.affine_weight + self.eps)

        x = x * self.stdev
        x = x + self.mean

        return x