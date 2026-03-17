import torch
import torch as t
from ANN.models.resnet_cifar10 import BasicBlockCifar
from QANN.operators.quanConv2d import QuanConv2dFuseBN
from QANN.operators.quanLinear import QuanLinear
from QANN.quantization.quantizer import LsqQuan, LsqQuanAct
from QANN.operators.quanAvgPool import QuanAvgPool
from QANN.operators.quanAddition import AdditionQuan

index2 = 0
def quantized_train_model_fusebn(model,weightBit, actBit):
    global index2
    children = list(model.named_children())
    for name, child in children:
        is_need = False
        if isinstance(child, BasicBlockCifar):
            model._modules[name].conv1 = QuanConv2dFuseBN(m=child.conv1, is_first=(index2==0), quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_a_fn=LsqQuanAct(bit=actBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_out_fn=LsqQuanAct(bit=actBit,all_positive=False,symmetric=False,per_channel=False))
            index2 = index2 + 1
            model._modules[name].conv2 = QuanConv2dFuseBN(m=child.conv2, is_first=(index2==0), quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_a_fn=LsqQuanAct(bit=actBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_out_fn=LsqQuanAct(bit=actBit,all_positive=False,symmetric=False,per_channel=False))
            index2 = index2 + 1
            if child.downsample is not None:
                child.downsample[0] = QuanConv2dFuseBN(m=child.downsample[0], is_first=(index2==0), quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_a_fn=LsqQuanAct(bit=actBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_out_fn=model._modules[name].conv2.quan_out_fn)
                child.downsample[0].is_init = True
                index2 = index2 + 1
            # else:
            #     child.downsample = model._modules[name].conv2.quan_out_fn
            child.ResidualAdd = AdditionQuan(quan_a_fn=LsqQuanAct(bit=actBit,all_positive=False,symmetric=False,per_channel=False))            
            is_need = True
        
        elif isinstance(child, t.nn.Linear):
            model._modules[name] = QuanLinear(m=child, quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
                                            quan_out_fn=LsqQuanAct(bit=actBit,all_positive=False,symmetric=False,per_channel=False))
            index2 = index2 + 1
            is_need = True
        elif isinstance(child, t.nn.Conv2d):
            if index2 == 0:
                model._modules[name] = QuanConv2dFuseBN(m=child, is_first=True, quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_a_fn=LsqQuanAct(bit=(actBit),all_positive=False,symmetric=False,per_channel=False),
                                                quan_out_fn=LsqQuanAct(bit=actBit,all_positive=False,symmetric=False,per_channel=False))
            else:
                model._modules[name] = QuanConv2dFuseBN(m=child, is_first=False, quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_out_fn=LsqQuanAct(bit=actBit,all_positive=False,symmetric=False,per_channel=False))
            index2 = index2 + 1
            is_need = True
        elif isinstance(child, t.nn.AvgPool2d) or isinstance(child, t.nn.AdaptiveAvgPool2d):
            model._modules[name] = QuanAvgPool(child, quan_out_fn=LsqQuanAct(bit=actBit,all_positive=False,symmetric=False,per_channel=False))
            index2 = index2 + 1
            is_need = True
        if not is_need:
            quantized_train_model_fusebn(child,weightBit, actBit)
