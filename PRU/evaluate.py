import nibabel as nib 
import bratsUtils
import os
pdpath = '../../exp/prediction/15161688626888067130_ebest'
gtpath = ''
def getnames(pdpath):
    pds = os.listdir(pdpath)
    names=
    for pd in pds:
        name = os.splitext(os.path.splitext(pds)[0])[0]