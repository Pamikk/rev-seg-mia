import nibabel as nib 
import os
import numpy as np
pdpath = '../../exp/prediction/15161688626888067130_ebest/'
gtpath = '../../../dataset/Mydataset/test'
def softDice(pred, target, smoothing=1, nonSquared=False):
    intersection = (pred * target).sum()
    if nonSquared:
        union = (pred).sum() + (target).sum()
    else:
        union = (pred * pred).sum() + (target * target).sum()
    dice = (2 * intersection + smoothing) / (union + smoothing)

    return dice.mean()
def getnames(pdpath):
    pds = os.listdir(pdpath)
    names=[]
    for pd in pds:
        name = os.path.splitext(os.path.splitext(pd)[0])[0]
        names.append(name)
    return names
def trans2gtshape(t,pred):
    s = pred.shape
    if len(s)>len(t):
        pred = pred.squeeze()
    newp = np.zeros(t)
    newp[:min(s[0],t[0]),:min(s[1],t[1]),:min(s[2],t[2])] = pred[:min(s[0],t[0]),:min(s[1],t[1]),:min(s[2],t[2])]
    return newp
def cal_single_dice(gt,pd):
    A = pd == 1
    P = pd == 2
    Amask = gt == 1
    Pmask = gt == 2
    Adice = softDice(A, Amask, 0, True)
    Pdice = softDice(P, Pmask, 0, True)
    return Adice,Pdice

def cal_all_dice(gtpath,pdpath,names):
    diceA,diceP=[],[]
    for name in names:
        pd = nib.load(os.path.join(pdpath,name+'.nii.gz'))
        pd = pd.get_fdata()
        gt = nib.load(os.path.join(gtpath,name,name+'_seg.nii.gz'))
        gt = gt.get_fdata()
        pd = trans2gtshape(gt.shape,pd)
        A,P = cal_single_dice(gt,pd)
        diceA.append(A)
        diceP.append(P)
    return diceA,diceP
names = getnames(pdpath)
diceA,diceP=cal_all_dice(gtpath,pdpath,names)
print(np.mean(diceA),np.mean(diceP))



