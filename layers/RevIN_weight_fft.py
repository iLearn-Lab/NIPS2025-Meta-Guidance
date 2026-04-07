import torch
import torch.nn as nn
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    '''
    def __init__(self, enc_in, seq_len, hidden_dims = [16, 16], hidden_layers = 2, kernel_size=3):
        super(Projector, self).__init__()
        
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=False)

        layers = [nn.Linear(enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers-1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]
        
        layers += [nn.Linear(hidden_dims[-1], 1, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.series_conv(x)
        x = x.view(batch_size, -1)
        y = self.backbone(x)
        return y

class FourierFilter(nn.Module):
    """
    Fourier Filter: to time-variant and time-invariant term
    """
    def __init__(self, mask_spectrum=None):
        super(FourierFilter, self).__init__()
        # self.mask_spectrum = mask_spectrum
        
    def forward(self, x):
        xf = torch.fft.rfft(x, dim=1)
        mask = torch.ones_like(xf)
        # mask[:, self.mask_spectrum, :] = 0
        mask[:, (int)(mask.shape[1]/4), :] = 0
        x_high = torch.fft.irfft(xf*mask, dim=1)
        x_low = x - x_high
        
        return x_low, x_high


class RevIN_weight(nn.Module):
    def __init__(self, configs):
        super(RevIN_weight, self).__init__()

        self.configs = configs
        self.num_features = configs.enc_in
        self.device = configs.gpu
        self.eps = 1e-5
        self.weight_rev = configs.weight_rev
        self.sqrtpi = np.sqrt(2*np.pi).item()
        self._init_params()
        print("\nfft RevIN active! weight_rev=", self.weight_rev)
        self.cnt = 0
    

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features).unsqueeze(0).unsqueeze(0))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features).unsqueeze(0).unsqueeze(0))
        self.after_x_multi = nn.Parameter(torch.ones(self.num_features).unsqueeze(0).unsqueeze(0))
        self.after_x_bias = nn.Parameter(torch.zeros(self.num_features).unsqueeze(0).unsqueeze(0))
        self.multi_learner = Projector(enc_in=self.num_features, seq_len=self.configs.seq_len)
        self.filter = FourierFilter()

    def forward(self, x, mode:str, mask):
        if mode == 'norm':
            x = self._normalize(x, mask)
        
        elif mode == 'denorm':
            x = self._denormalize(x)
        
        else: raise NotImplementedError

        return x

    


    def SelectRange(self, x, minn, maxx, pdim):
        # x是一个2维tensor，去除x第二维的第pdim索引数字超过minn和maxx的第一维元素
        ind = (((x[:, pdim] < minn) | (x[:, pdim] >= maxx))).nonzero()
        ind = torch.unique(ind)
        ind_r = torch.arange(x.shape[0]).to(self.device)
        ind_r = ind_r[~torch.isin(ind_r, ind)]
        result_tensor = torch.index_select(x, dim=0, index=ind_r)
        return result_tensor

    def NormalDistribution(self, py, multi, sigma):
        # 返回 N(0,sigma)正态分布在py处的值
        return multi * np.exp(-1 * py*py/(2*sigma*sigma)) / (self.sqrtpi * sigma)
        # return self.multi * torch.exp(-1 * x*x/(2*self.sigma*self.sigma)) / (np.sqrt(2*np.pi) * self.sigma)

    
    def _get_weight(self, x, mask):
        # 对数据段x，遮罩mask，生成一次权重，每个mask==0周围的值获得正态分布权重
        # 先对数据进行归一化
        x_d = x.clone().detach()
        x_d = torch.nn.functional.normalize(x_d, dim=1)

        multi = self.multi_learner(x_d)[:, 0].exp()
        # sigma = self.sigma_learner(x_d)[:, 0].exp()
        sigma = 1

        self.cnt = self.cnt + 1
        if(self.cnt > 200):
            print("multi", multi[0])
            # print("sigma", sigma[0])
            self.cnt = 0
        
        weight = torch.ones_like(x).to(self.device)
        pos = (mask == 0).nonzero()

        for i in range(-3, 4):
            pos_py = pos.clone()
            pos_py[:, 1] += i
            pos_py = self.SelectRange(pos_py, 0, x.shape[1], 1)
            # weight[pos_py[:, 0], pos_py[:, 1], pos_py[:, 2]] += self.NormalDistribution(i, multi[pos_py[:, 0]], sigma[pos_py[:, 0]])
            weight[pos_py[:, 0], pos_py[:, 1], pos_py[:, 2]] += self.NormalDistribution(i, multi[pos_py[:, 0]], sigma)

        return weight

    def _normalize(self, x, mask):
        
        

        x_low, x_high = self.filter(x)

        # ************** 对low进行权重RevIN *************
        if self.weight_rev:
            weight = self._get_weight(x_low, mask)
        else:
            weight = mask

        self.weight = weight

        x_d = x_low * weight
        cnt = torch.sum(weight, dim=1)
        cnt[cnt == 0] = 1

        mean = torch.sum(x_d, dim=1) / cnt
        mean = mean.unsqueeze(1)
        self.mean = mean
        
        x_low = x_low - self.mean
        x_low = x_low.masked_fill(mask == 0, 0)
        x_d = x_low * weight


        stdev = torch.sqrt(torch.sum(x_d * x_d, dim=1) / cnt + self.eps)
        stdev = stdev.unsqueeze(1)
        self.stdev = stdev

        x_low = x_low / self.stdev
        # x_low = ((x_low * self.weight) * self.after_x_multi + self.after_x_bias) if self.weight_rev else x_low

        # ************** 对high进行普通RevIN *************
        x_high_mask = x_high.clone().detach() * mask
        self.mean_high = (torch.sum(x_high_mask, dim=1) / torch.sum(mask, dim=1)).unsqueeze(1).detach() # B x 1 x E
        x_high_mask -= self.mean_high
        self.std_high = (torch.sqrt(torch.sum(x_high_mask * x_high_mask, dim=1) / torch.sum(mask, dim=1) + self.eps)).unsqueeze(1).detach()
        
        x_high = (x_high-self.mean_high) / self.std_high
        # self.mean_high = x_high.mean(1, keepdim=True).detach() # B x 1 x E
        # x_high = x_high - self.mean_high
        # self.std_high = torch.sqrt(torch.var(x_high, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        # x_high = x_high / self.std_high

        x = x_low + x_high

        # 调整参数
        x = x * self.affine_weight
        x = x + self.affine_bias
        return x * mask

    def _denormalize(self, x):

        # 调整参数
        x = x - self.affine_bias
        x = x / (self.affine_weight + self.eps*self.eps)

        x_low, x_high = self.filter(x)

        # ************** 对low进行反权重RevIN *************
        self.weight = self.weight.masked_fill(self.weight == 0, 1)
        # x_low = ((x_low - self.after_x_bias) / self.after_x_multi / self.weight) if self.weight_rev else x_low
        x_low = x_low * self.stdev
        x_low = x_low + self.mean

        # ************** 对high进行普通RevIN *************
        x_high = x_high * self.std_high + self.mean_high

        x = x_low + x_high
        return x
