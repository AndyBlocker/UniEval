import torch
import torch as t

class QuanAvgPool(t.nn.Module):
    def __init__(self,m, quan_out_fn):
        super(QuanAvgPool,self).__init__()
        assert isinstance(m, t.nn.AvgPool2d) or isinstance(m, t.nn.AdaptiveAvgPool2d), "average pooling!!!"
        self.m = m
        self.quan_out_fn = quan_out_fn
        self.is_init = False
    def forward(self,x):
        if self.is_init == False:
            # print("in QuanAvgPool input_mean:",x.abs().mean().item())
            x = self.m(x)
            self.quan_out_fn.init_from(x,weight=False)
            self.is_init = True
            # print("in QuanAvgPool output_mean:",x.abs().mean().item())
            return x

        x = self.m(x)
        x = self.quan_out_fn(x)
        # print("train AvgPool output:",(x/self.quan_out_fn.s).abs()[0,0,0,:])
        return x
