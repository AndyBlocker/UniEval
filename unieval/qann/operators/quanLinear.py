import torch
import torch as t

class QuanLinear(t.nn.Linear):
    def __init__(self, m: t.nn.Linear, quan_w_fn=None, quan_out_fn=None):
        assert type(m) == t.nn.Linear
        super().__init__(m.in_features, m.out_features,
                         bias=True if m.bias is not None else False)
        self.quan_w_fn = quan_w_fn
        self.quan_out_fn = quan_out_fn
        self.m = m

        self.weight = t.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight)
        # self.quan_out_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())
        else:
            self.bias = None

        self.is_init = False
        # self.l1_loss = 0
        self.l2_loss = 0.0
        self.absoluteValue = 0.0
        
    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)

        # Weight-only quantization path (no quan_out_fn)
        if self.quan_out_fn is None:
            return t.nn.functional.linear(x, quantized_weight, self.bias)

        if self.is_init == False:
            bias = self.bias.reshape(1, -1) if self.bias is not None else 0.0
            out = t.nn.functional.linear(x, self.weight, bias=None) + bias
            self.quan_out_fn.init_from(out, weight=False)
            self.is_init = True
            return out

        out = t.nn.functional.linear(x, quantized_weight, bias=None)
        quantized_out = self.quan_out_fn(out, clip=False)
        if self.bias is not None:
            quantized_bias = self.quan_out_fn(self.bias).reshape(1, -1)
            quantized_out = quantized_out + quantized_bias
        quantized_out = torch.clip(quantized_out, min=self.quan_out_fn.s * self.quan_out_fn.thd_neg, max=self.quan_out_fn.s * self.quan_out_fn.thd_pos)
        return quantized_out
