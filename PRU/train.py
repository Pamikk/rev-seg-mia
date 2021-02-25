import comet_ml
import torch
import bratsDataset
import segmenter
import systemsetup
import argparse
import experiments
import experiments.noNewReversibleDia as dia
import experiments.noNewReversibleDiaCom as diaCom
import experiments.noNewReversibleFat as rev
import experiments.mybaseline as base
import experiments.noNewReversibleFatBCE as BCE
import experiments.noNewReversibleMSE as MSE
import experiments.noNewReversibleFatCom as com
import experiments.noNewReversibleFatBn as bn
import experiments.noNewReversibleTopK as TopK
exps={'base':base,'rev':rev,'Dia':dia,'DiaCom':diaCom,'BCE':BCE,'Com':com,'MSE':MSE,'bn':bn,'topk':TopK}
class bcolors:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'

def main(args):
    expConfig=exps[args.exp]
    # setup experiment logging to comet.ml
    if expConfig.LOG_COMETML:
        hyper_params = {"experimentName": expConfig.EXPERIMENT_NAME,
                        "epochs": expConfig.EPOCHS,
                        "batchSize": expConfig.BATCH_SIZE,
                        "channels": expConfig.CHANNELS,
                        "virualBatchsize": expConfig.VIRTUAL_BATCHSIZE}
        expConfig.experiment.log_(hyper_params)
        expConfig.experiment.add_tags([expConfig.EXPERIMENT_NAME, "ID{}".format(expConfig.id)])
        if hasattr(expConfig, "EXPERIMENT_TAGS"): expConfig.experiment.add_tags(expConfig.EXPERIMENT_TAGS)
        print(bcolors.OKGREEN + "Logging to comet.ml" + bcolors.ENDC)
    else:
        print(bcolors.WARNING + "Not logging to comet.ml" + bcolors.ENDC)

    # log parameter count
    if expConfig.LOG_PARAMCOUNT:
        paramCount = sum(p.numel() for p in expConfig.net.parameters() if p.requires_grad)
        print("Parameters: {:,}".format(paramCount).replace(",", "'"))

    #load data
    randomCrop = None
    trainset = bratsDataset.BratsDataset(systemsetup.BRATS_PATH, expConfig, mode="train", randomCrop=randomCrop)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=expConfig.BATCH_SIZE, shuffle=True, pin_memory=False)

    valset = bratsDataset.BratsDataset(systemsetup.BRATS_PATH, expConfig, mode="validation")
    valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, pin_memory=False)

    challengeValset = bratsDataset.BratsDataset(systemsetup.BRATS_VAL_PATH, expConfig,hasMasks=False, mode="validation", returnOffsets=True)
    challengeValloader = torch.utils.data.DataLoader(challengeValset, batch_size=1, shuffle=False, pin_memory=True)

    seg = segmenter.Segmenter(expConfig, trainloader, valloader, challengeValloader)#,trainvalloader)
    if hasattr(expConfig, "VALIDATE_ALL") and expConfig.VALIDATE_ALL:
        seg.validateAllCheckpoints()
    elif hasattr(expConfig, "PREDICT") and expConfig.PREDICT:
        seg.makePredictions()
    else:
        seg.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PRU Training')
    parser.add_argument('--exp', default='base', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    main(parser.parse_args())