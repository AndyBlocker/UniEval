import torch



ANN = ResNet20()

QANN = A2QConverter(ANN)

SNN = SNNWrapper(QANN)

evalute_state = SNNEvaluator(SNN)

print(evalute_state)

