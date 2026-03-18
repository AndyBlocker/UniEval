import torch
import torch.nn as nn

class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()

    def forward(self, x):
        return x


def fuse(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    beta = bn.weight
    gamma = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean)/var_sqrt * beta + gamma

    # fused_conv = nn.Conv2d(
    #     conv.in_channels,
    #     conv.out_channels,
    #     conv.kernel_size,
    #     conv.stride,
    #     conv.padding,
    #     conv.dilation,
    #     conv.groups,
    #     bias=True,
    #     padding_mode=conv.padding_mode
    # )
    conv.weight = nn.Parameter(w)
    conv.bias = nn.Parameter(b)
    return conv

def fuse_module(m):
    children = list(m.named_children())
    conv = None
    conv_name = None

    for name, child in children:
        if (isinstance(child, nn.BatchNorm2d) or isinstance(child, nn.SyncBatchNorm)) and conv:
            bc = fuse(conv, child)
            m._modules[conv_name] = bc
            m._modules[name] = DummyModule()
            conv = None
        elif isinstance(child, nn.Conv2d):
            conv = child
            conv_name = name
        else:
            fuse_module(child)

def fuse_for_train(conv, child: nn.BatchNorm2d):
    conv.beta = child.bias
    conv.gamma = child.weight
    conv.running_mean = child.running_mean
    conv.running_var = child.running_var    
    return conv
    

def fuse_module_train(m):
    children = list(m.named_children())
    conv = None
    conv_name = None

    for name, child in children:
        if (isinstance(child, nn.BatchNorm2d) or isinstance(child, nn.SyncBatchNorm)) and conv:
            bc = fuse_for_train(conv, child)
            m._modules[conv_name] = bc
            m._modules[name] = DummyModule()
            conv = None
        elif isinstance(child, nn.Conv2d):
            conv = child
            conv_name = name
        else:
            fuse_module_train(child)



def validate(net, input_, cuda=True):
    net.eval()
    if cuda:
        input_ = input_.cuda()
        net.cuda()
    # import time
    # s = time.time()
    a = net(input_)
    print(a)
    if cuda:
        torch.cuda.synchronize()
        torch.set_default_dtype(torch.float64)
        net.double()
    # print(time.time() - s)
    fuse_module(net)
    # print(mbnet)
    # s = time.time()
    if cuda:
        torch.cuda.synchronize()
        torch.set_default_dtype(torch.float32)
        net.float()
    b = net(input_)
    print(b)
    # print(time.time() - s)
    return (a - b).abs().max().item()

if __name__ == '__main__':
    import torchvision
    import timm
    mbnet = torch.load("/home/kang_you/LLconverter/output/cifar10_vgg16_quan4_1677759303845/ann_model_best.pth")
    # myquan_replace(mbnet)
    # torch.set_default_dtype(torch.float64)
    # mbnet.double()
    print(mbnet)
    mbnet.eval()
    print(validate(mbnet, torch.randn(256, 3, 32, 32), False))