import nibabel as nib 
import os
import numpy as np
import argparse
import medpy.metric.binary as medpyMetrics
gtpath = '../../../dataset/Mydataset/labels'
exps={'base':'../../exp/prediction/Baseline1/',
'rev':'../../exp/prediction/Reversible1',
'BCE':'../../exp/prediction/ReversibleBCE1',
'Com':'../../exp/prediction/ReversibleCom1',
'Dia':"../../exp/prediction/ReversibleDia1",
'topk':"../../exp/prediction/ReversibleTopk1",
'DiaCom':"../../exp/prediction/ReversibleDiaCom1"}
def getNSD(pred,target):
    surDist1 = medpyMetrics.__surface_distances(pred, target)
    surDist2 = medpyMetrics.__surface_distances(target, pred)
    hd95 = np.percentile(np.hstack((surDist1, surDist2)), 95)
    return hd95
    
def softDice(pred, target, smoothing=1, nonSquared=False):
    intersection = (pred * target).sum()
    union = (pred).sum() + (target).sum()
    dice = (2 * intersection) / (union)
    return dice.mean()
def softJI(pred, target, smoothing=1, nonSquared=False):
    intersection = (pred * target).sum()
    union = (pred).sum() + (target).sum()
    dice = (intersection) / (union-intersection)
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
def cal_single_NSD(gt,pd):
    A = pd == 1
    P = pd == 2
    Amask = gt == 1
    Pmask = gt == 2
    Adice = getNSD(A, Amask)
    Pdice = getNSD(P, Pmask)
    return Adice,Pdice
def cal_single_JI(gt,pd):
    A = pd == 1
    P = pd == 2
    Amask = gt == 1
    Pmask = gt == 2
    Adice = softJI(A, Amask, 0, True)
    Pdice = softJI(P, Pmask, 0, True)
    return Adice,Pdice
def cal_all_dice(gtpath,pdpath,names):
    diceA,diceP,avg=[],[],[]
    for name in names:
        pd = nib.load(os.path.join(pdpath,name+'.nii.gz'))
        pd = pd.get_fdata()
        gt = nib.load(os.path.join(gtpath,name+'_seg.nii.gz'))
        gt = gt.get_fdata()
        pd = trans2gtshape(gt.shape,pd)
        A,P = cal_single_dice(gt,pd)
        diceA.append(A)
        diceP.append(P)
        avg.append((A+P)/2)
    return diceA,diceP,avg
def cal_all_JI(gtpath,pdpath,names):
    diceA,diceP,avg=[],[],[]
    for name in names:
        pd = nib.load(os.path.join(pdpath,name+'.nii.gz'))
        pd = pd.get_fdata()
        gt = nib.load(os.path.join(gtpath,name+'_seg.nii.gz'))
        gt = gt.get_fdata()
        pd = trans2gtshape(gt.shape,pd)
        A,P = cal_single_JI(gt,pd)
        diceA.append(A)
        diceP.append(P)
        avg.append((A+P)/2)
    return diceA,diceP,avg
def cal_all_NSD(gtpath,pdpath,names):
    diceA,diceP,avg=[],[],[]
    for name in names:
        pd = nib.load(os.path.join(pdpath,name+'.nii.gz'))
        pd = pd.get_fdata()
        gt = nib.load(os.path.join(gtpath,name+'_seg.nii.gz'))
        gt = gt.get_fdata()
        pd = trans2gtshape(gt.shape,pd)
        A,P = cal_single_NSD(gt,pd)
        diceA.append(A)
        diceP.append(P)
        avg.append((A+P)/2)
    return diceA,diceP,avg
def main(args):
    pdpath = exps[args.exp]
    names = getnames(pdpath)
    diceA,diceP,avg=cal_all_dice(gtpath,pdpath,names)
    val1=np.mean(diceA)
    val2=np.mean(diceP)
    val = np.mean(avg)
    print("{}\t{}\t{}".format(val*100,val1*100,val2*100))
    diceA,diceP,avg=cal_all_NSD(gtpath,pdpath,names)
    val1=np.mean(diceA)
    val2=np.mean(diceP)
    val = np.mean(avg)
    print("{}\t{}\t{}".format(val,val1,val2))
    diceA,diceP,avg=cal_all_JI(gtpath,pdpath,names)
    val1=np.mean(diceA)
    val2=np.mean(diceP)
    val = np.mean(avg)
    print("{}\t{}\t{}".format(val,val1,val2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PRU Training')
    parser.add_argument('--exp', default='base', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    main(parser.parse_args())



