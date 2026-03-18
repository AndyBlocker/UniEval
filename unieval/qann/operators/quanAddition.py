import torch
import torch as t

class AdditionQuan(t.nn.Module):
    def __init__(
        self,
        quan_a_fn,
    ):
        super().__init__()
        self.quan_a_fn = quan_a_fn
        self.is_init = False
        self.thd_pos = quan_a_fn.thd_pos
        self.thd_neg = quan_a_fn.thd_neg
    
    def forward(self,x1,x2):
        if self.is_init == False:
            x = x1 + x2
            self.quan_a_fn.init_from(x,weight=False)
            self.is_init = True
            return x
        else:
            # print("AdditionQuan input1:",(x1).abs().mean(),"input2:",x2.abs().mean())
            out = self.quan_a_fn(x1+x2)
            # print("AdditionQuan Output:",out.abs().mean())
        return out
