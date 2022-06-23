import torch.nn.functional as F


def DCE(inputs, targets, smooth=1):
    inputs = F.sigmoid(inputs)

    # flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice = (2. * intersection+smooth) / (inputs.sum()+targets.sum()+smooth)
    return 1-dice




def  FocalTverskyLoss(inputs, targets, smooth=1, alpha=0.7, beta=0.3, gamma=(4/3)):
    # comment out if your model contains a sigmoid or equivalent activation layer
    inputs = F.sigmoid(inputs)

    # flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    # True Positives, False Positives & False Negatives
    TP = (inputs * targets).sum()
    FP = ((1-targets) * inputs).sum()
    FN = (targets * (1-inputs)).sum()

    Tversky = (TP+smooth) / (TP+alpha * FP+beta * FN+smooth)
    FocalTversky = (1-Tversky) ** gamma

    return FocalTversky
