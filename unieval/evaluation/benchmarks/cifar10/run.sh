python /home/kang_you/UniEval/unieval/evaluation/benchmarks/cifar10/trainQAT.py --WeightBit 4 --ActBit 4 \
 --ann-ckpt /home/kang_you/UniEval/unieval/evaluation/benchmarks/cifar10/runs/resnet20_cifar10/ \
 --resume /home/kang_you/UniEval/runs/resnet20_cifar10_qat/best.pt \
 --data-dir /data/cifar-10-python --lr 1e-4 --eval-only

