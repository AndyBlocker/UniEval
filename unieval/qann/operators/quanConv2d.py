import torch
import torch as t


def reshape_to_weight(x: torch.Tensor) -> torch.Tensor:
    """
    将 1D 的 BN 缩放系数 reshape 成可与 Conv2d 权重广播相乘的形状。

    - x: shape = [out_channels]
    - return: shape = [out_channels, 1, 1, 1]
    """

    return x.reshape(-1, 1, 1, 1)


def reshape_to_bias(x: torch.Tensor) -> torch.Tensor:
    """
    将 bias/偏置相关向量 reshape 成标准的一维向量，便于后续加法广播。
    """

    return x.reshape(-1)


class QuanConv2dFuseBN(t.nn.Conv2d):
    def __init__(self, m: t.nn.Conv2d, quan_w_fn=None, quan_a_fn=None, quan_out_fn=None,momentum=0.9, eps=1e-5,is_first=False):
        assert type(m) == t.nn.Conv2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.m = m
        self.quan_w_fn = quan_w_fn
        self.quan_out_fn = quan_out_fn
        self.quan_a_fn = quan_a_fn
        
        self.momentum = momentum
        self.eps = eps
        self.register_parameter('beta', t.nn.Parameter(torch.zeros(m.out_channels)))
        self.register_parameter('gamma', t.nn.Parameter(torch.ones(m.out_channels)))
        self.register_buffer('running_mean', torch.zeros(m.out_channels))
        self.register_buffer('running_var', torch.ones(m.out_channels))

        self.weight = t.nn.Parameter(m.weight.detach())
        # print(self.weight.mean())
        self.quan_w_fn.init_from(m.weight)
        # self.quan_out_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())
        else:
            self.bias = None
        self.is_init = False
        self.is_first = is_first
        # if self.is_first:
        #     self.quan_a_fn.init_from(m.weight)
        # self.l1_loss = 0
        self.l2_loss = 0
        self.absoluteValue = 0

    def forward(self, x):

        # print("QuanConv2dFuseBN Input",x.abs().mean())        
        running_std = torch.sqrt(self.running_var + self.eps)
        weight = self.weight * reshape_to_weight(self.gamma / running_std)

        # print("QuanConv2dFuseBN self.bias", self.bias)
        if self.bias is not None:
            bias = self.bias * self.gamma / running_std + reshape_to_bias(self.beta - self.gamma * self.running_mean / running_std)
        else:
            bias = reshape_to_bias(self.beta - self.gamma * self.running_mean / running_std)

        quantized_weight = self.quan_w_fn(weight)

        if self.is_init == False:
            if self.is_first:
                self.quan_a_fn.init_from(x,weight=False)
            out = self._conv_forward(x, weight,bias = None) + bias.reshape(1,-1,1,1)
            self.quan_out_fn.init_from(out,weight=False)
            self.is_init = True
            return out
        
        out = self._conv_forward(x, quantized_weight,bias = None)        
        quantized_bias = self.quan_out_fn(bias)
        # print("QuanConv2dFuseBN _conv_forward",out.abs().mean(), "quantized_weight", quantized_weight.abs().mean(),"quantized_bias",quantized_bias.abs().mean())        

        # quantized_bias = bias
        # print("QuanConv2dFuseBN","x",x.abs().mean(), "quantized_weight", quantized_weight.abs().mean(),"out",out.abs().mean(),self.quan_out_fn)
        quantized_out = torch.clip(self.quan_out_fn(out,clip=False) + quantized_bias.reshape(1,-1,1,1),min=self.quan_out_fn.s*self.quan_out_fn.thd_neg,max=self.quan_out_fn.s*self.quan_out_fn.thd_pos)
        # print("QuanConv2dFuseBN Output",torch.nn.functional.relu(quantized_out).abs().mean())
        return quantized_out
