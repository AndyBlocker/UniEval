python /home/kang_you/UniEval/unieval/evaluation/benchmarks/cifar10/snnInfer.py --WeightBit 4 --ActBit 4 --time-step 32 \
 --ann-ckpt /home/kang_you/UniEval/unieval/evaluation/benchmarks/cifar10/runs/resnet20_cifar10 \
 --qann-ckpt /home/kang_you/UniEval/runs/resnet20_cifar10_qat/ \
 --data-dir /data/cifar-10-python --lr 1e-4

