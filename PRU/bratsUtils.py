import medpy.metric.binary as medpyMetrics
import numpy as np
import math
import torch

def softDice(pred, target, smoothing=1, nonSquared=False):
    intersection = (pred * target).sum(dim=(1, 2, 3))
    if nonSquared:
        union = (pred).sum() + (target).sum()
    else:
        union = (pred * pred).sum(dim=(1, 2, 3)) + (target * target).sum(dim=(1, 2, 3))
    dice = (2 * intersection + smoothing) / (union + smoothing)

    #fix nans
    dice[dice != dice] = dice.new_tensor([1.0])

    return dice.mean()

def dice(pred, target):
    predBin = (pred > 0.5).float()
    return softDice(predBin, target, 0, True).item()

def diceLoss(pred, target, nonSquared=False):
    return 1 - softDice(pred, target, nonSquared=nonSquared)

def bratsDiceLoss(outputs, labels, nonSquared=False):

    #bring outputs into correct shape
    A,P = outputs.chunk(2, dim=1)
    s = A.shape
    A = A.view(s[0], s[2], s[3], s[4])
    P = P.view(s[0], s[2], s[3], s[4])

    # bring masks into correct shape

    Amask,Pmask = labels.chunk(2, dim=1)
    s = Amask.shape
    Amask = Amask.view(s[0], s[2], s[3], s[4])
    Pmask = Pmask.view(s[0], s[2], s[3], s[4])

    #calculate losses
    ALoss = diceLoss(A, Amask, nonSquared=nonSquared)
    PLoss = diceLoss(P, Pmask, nonSquared=nonSquared)
    return (PLoss + ALoss) / 5

def bratsDiceLossOriginal5(outputs, labels, nonSquared=False):
    outputList = list(outputs.chunk(5, dim=1))
    labelsList = list(labels.chunk(5, dim=1))
    totalLoss = 0
    for pred, target in zip(outputList, labelsList):
        totalLoss = totalLoss + diceLoss(pred, target, nonSquared=nonSquared)
    return totalLoss


def sensitivity(pred, target):
    predBin = (pred > 0.5).float()
    intersection = (predBin * target).sum()
    allPositive = target.sum()

    # special case for zero positives
    if allPositive == 0:
        return 1.0
    return (intersection / allPositive).item()

def specificity(pred, target):
    predBinInv = (pred <= 0.5).float()
    targetInv = (target == 0).float()
    intersection = (predBinInv * targetInv).sum()
    allNegative = targetInv.sum()
    return (intersection / allNegative).item()

def getHd95(pred, target):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    if np.count_nonzero(pred) > 0 and np.count_nonzero(target):
        surDist1 = medpyMetrics.__surface_distances(pred, target)
        surDist2 = medpyMetrics.__surface_distances(target, pred)
        hd95 = np.percentile(np.hstack((surDist1, surDist2)), 95)
        return hd95
    else:
        # Edge cases that medpy cannot handle
        return -1

def getAMask(labels):
    return (labels ==1).float()


def getPMask(labels):
    return (labels == 2).float()
