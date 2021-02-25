import torch
import torch.nn as nn
import time
import bratsUtils
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import os
import dataProcessing.utils as utils
import systemsetup
from tqdm import tqdm
class Segmenter:

    def __init__(self, expConfig, trainDataLoader, valDataLoader, challengeValDataLoader):#,trainvalDataLoader):
        self.expConfig = expConfig
        self.trainDataLoader = trainDataLoader
        self.valDataLoader = valDataLoader
        #self.trainvalDataLoader = trainvalDataLoader
        self.challengeValDataLoader = challengeValDataLoader
        self.experiment = expConfig.experiment
        self.checkpointsBasePathLoad = systemsetup.CHECKPOINT_BASE_PATH
        self.checkpointsBasePathSave= systemsetup.CHECKPOINT_BASE_PATH
        self.predictionsBasePath = systemsetup.PREDICTIONS_BASE_PATH
        self.startFromEpoch = 0

        self.bestMeanDice = 0
        self.bestMeanDiceEpoch = 0

        self.movingAvg = 0
        self.bestMovingAvg = 0
        self.bestMovingAvgEpoch = 1e9
        self.EXPONENTIAL_MOVING_AVG_ALPHA = 0.95
        self.EARLY_STOPPING_AFTER_EPOCHS = 100
        

        # restore model if requested
        if hasattr(expConfig, "RESTORE_ID") and hasattr(expConfig, "RESTORE_EPOCH"):
            self.startFromEpoch = self.loadFromDisk(expConfig.RESTORE_ID, expConfig.RESTORE_EPOCH) + 1
            self.expConfig.id = expConfig.RESTORE_ID
            print("Loading checkpoint with id {} at epoch {}".format(expConfig.RESTORE_ID, self.startFromEpoch-1))

        # Run on GPU or CPU
        if torch.cuda.is_available():
            print("using cuda (", torch.cuda.device_count(), "device(s))")
            if torch.cuda.device_count() > 1:
                expConfig.net = nn.DataParallel(expConfig.net)
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            print("using cpu")
        expConfig.net = expConfig.net.to(self.device)
        
        torch.cuda.empty_cache() 
        basePath = self.checkpointsBasePathSave + "{}".format(self.expConfig.id)
        if not os.path.exists(basePath):
            os.makedirs(basePath)
        self.trainlog = open(os.path.join(basePath,'train_log.txt'),'a+')
        self.vallog = open(os.path.join(basePath,'val_log.txt'),'a+')

    def validateAllCheckpoints(self):

        expConfig = self.expConfig

        print('==== VALIDATING ALL CHECKPOINTS ====')
        print(self.expConfig.EXPERIMENT_NAME)
        print("ID: {}".format(expConfig.id))
        print("RESTORE ID {}".format(expConfig.RESTORE_ID))
        print('====================================')

        for epoch in range(self.startFromEpoch, self.expConfig.EPOCHS):
            self.loadFromDisk(expConfig.RESTORE_ID, epoch)
            self.validate(epoch)

        #print best mean dice
        print("Best mean dice: {:.4f} at epoch {}".format(self.bestMeanDice, self.bestMeanDiceEpoch))

    def makePredictions(self):
        # model is already loaded from disk by constructor
        
        expConfig = self.expConfig
        assert(hasattr(expConfig, "RESTORE_ID"))
        assert(hasattr(expConfig, "RESTORE_EPOCH"))
        id = expConfig.RESTORE_ID
        epoch = expConfig.RESTORE_EPOCH

        print('============ PREDICTING ============')
        print(self.expConfig.EXPERIMENT_NAME)
        print("ID: {}".format(expConfig.id))
        print("RESTORE ID {}".format(expConfig.RESTORE_ID))
        print("RESTORE EPOCH {}".format(expConfig.RESTORE_EPOCH))
        print('====================================')
        
        basePath = os.path.join(self.predictionsBasePath, self.expConfig.EXPERIMENT_NAME)
        if not os.path.exists(basePath):
            os.makedirs(basePath)

        with torch.no_grad():
            for i, data in tqdm(enumerate(self.challengeValDataLoader)):
                inputs, pids, xOffset, yOffset, zOffset = data
                print("processing {}".format(pids[0]))
                inputs = inputs.to(self.device)

                #predict labels and bring into required shape
                outputs = expConfig.net(inputs)
                n,c,h,w,d = outputs.shape
                nh,nw,nd = h+xOffset,w+yOffset,d+zOffset
                fullsize = torch.zeros([n,c,nh,nw,nd])
                fullsize[:,:,xOffset:xOffset+h,yOffset:yOffset+w,zOffset:zOffset+d]=outputs

                #binarize output
                A,P = fullsize.chunk(2, dim=1)
                s = fullsize.shape
                A = (A > 0.5).view(s[2], s[3], s[4])
                P = (P > 0.5).view(s[2], s[3], s[4])

                result = fullsize.new_zeros((s[2], s[3], s[4]), dtype=torch.uint8)
                result[A] = 1
                result[P] = 2

                npResult = result.cpu().numpy()
                path = os.path.join(basePath, "{}.nii.gz".format(pids[0]))
                utils.save_nii(path, npResult, None, None)

        print("Done :)")

    def train(self):

        expConfig = self.expConfig
        expConfig.optimizer.zero_grad()
        
        print('======= RUNNING EXPERIMENT =======')
        print(self.expConfig.EXPERIMENT_NAME)
        print("ID: {}".format(expConfig.id))
        print('==================================')
        self.trainlog.write(self.expConfig.EXPERIMENT_NAME+'\n')

        # for epoch in range(self.startFromEpoch, self.expConfig.EPOCHS):
        epoch = self.startFromEpoch
        while epoch < self.expConfig.EPOCHS and epoch <= self.bestMovingAvgEpoch + self.EARLY_STOPPING_AFTER_EPOCHS:

            running_loss = 0.0
            startTime = time.time()

            # set net up training
            self.expConfig.net.train()
            num = len(self.trainDataLoader)
            for i, data in tqdm(enumerate(self.trainDataLoader)):

                #load data
                inputs, pid, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                #forward and backward pass
                outputs = expConfig.net(inputs)
                loss = expConfig.loss(outputs, labels)
                del inputs, outputs, labels
                loss.backward()

                #update params
                if i == len(self.trainDataLoader) - 1 or i % expConfig.VIRTUAL_BATCHSIZE == (expConfig.VIRTUAL_BATCHSIZE - 1):
                    expConfig.optimizer.step()
                    expConfig.optimizer.zero_grad()

                #logging every K iterations
                running_loss += loss.item()
                del loss
            
            #logging at end of epoch
            if expConfig.LOG_MEMORY_EVERY_EPOCH: self.logMemoryUsage()
            if expConfig.LOG_EPOCH_TIME:
                print("Time for epoch: {:.2f}s".format(time.time() - startTime))
            if expConfig.LOG_LR_EVERY_EPOCH:
                for param_group in expConfig.optimizer.param_groups:
                    print("Current lr: {:.6f}".format(param_group['lr']))
                    print(running_loss/num)
                    self.trainlog.write("{}\t{:.6f}\t{:.6f}\n".format(epoch,param_group['lr'],running_loss/num))
            #validation at end of epoch
            if epoch % expConfig.VALIDATE_EVERY_K_EPOCHS == expConfig.VALIDATE_EVERY_K_EPOCHS - 1:
                self.validate(epoch)
                #self.valtrain(epoch)
            self.trainlog.flush()
            self.vallog.flush()
            #take lr sheudler step
            if hasattr(expConfig, "lr_sheudler"):
                if isinstance(expConfig.lr_sheudler, optim.lr_scheduler.ReduceLROnPlateau):
                    expConfig.lr_sheudler.step(self.movingAvg)
                else:
                    expConfig.lr_sheudler.step()

            #save model
            if expConfig.SAVE_CHECKPOINTS:
                if epoch % expConfig.SAVE_EVERY_K_EPOCHS == expConfig.SAVE_EVERY_K_EPOCHS - 1:
                    self.saveToDisk(epoch)
                    torch.cuda.empty_cache() 

            epoch = epoch + 1

        #print best mean dice
        print("Best mean dice: {:.4f} at epoch {}".format(self.bestMeanDice, self.bestMeanDiceEpoch))
        self.saveToDisk(epoch)
        self.trainlog.close()
        self.vallog.close()
    def validate(self, epoch):

        #set net up for inference
        self.expConfig.net.eval()

        expConfig = self.expConfig
        hausdorffEnabled = (expConfig.LOG_HAUSDORFF_EVERY_K_EPOCHS > 0)
        logHausdorff = hausdorffEnabled and epoch % expConfig.LOG_HAUSDORFF_EVERY_K_EPOCHS == (expConfig.LOG_HAUSDORFF_EVERY_K_EPOCHS - 1)

        startTime = time.time()
        with torch.no_grad():
            diceA, diceP = [], []
            sensA, sensP = [], []
            specA, specP = [], []
            hdA, hdP = [], []
            #buckets = np.zeros(5)

            for i, data in enumerate(self.valDataLoader):

                # feed inputs through neural net
                inputs, _, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = expConfig.net(inputs)

                if expConfig.TRAIN_ORIGINAL_CLASSES:
                    outputsOriginal5 = outputs
                    outputs = torch.argmax(outputs, 1)
                    #hist, _ = np.histogram(outputs.cpu().numpy(), 5, (0, 4))
                    #buckets = buckets + hist
                    A = bratsUtils.getAMask(outputs)
                    P = bratsUtils.getPMask(outputs)

                    labels = torch.argmax(labels, 1)
                    AMask = bratsUtils.getAMask(labels)
                    PMask = bratsUtils.getPMask(labels)

                else:

                    #separate outputs channelwise
                    A,P = outputs.chunk(2, dim=1)
                    s = A.shape
                    A = A.view(s[0], s[2], s[3], s[4])
                    P = P.view(s[0], s[2], s[3], s[4])

                    AMask,PMask = labels.chunk(2, dim=1)
                    s = AMask.shape
                    AMask = AMask.view(s[0], s[2], s[3], s[4])
                    PMask = PMask.view(s[0], s[2], s[3], s[4])

                #TODO: add special evaluation metrics for original 5

                #get dice metrics
                diceA.append(bratsUtils.dice(A, AMask))
                diceP.append(bratsUtils.dice(P, PMask))

                #get sensitivity metrics
                sensA.append(bratsUtils.sensitivity(A, AMask))
                sensP.append(bratsUtils.sensitivity(P, PMask))

                #get specificity metrics
                specA.append(bratsUtils.specificity(A, AMask))
                specP.append(bratsUtils.specificity(P, PMask))

                #get hausdorff distance
                '''if logHausdorff:
                    lists = [hdA, hdP]
                    results = [A, P]
                    masks = [AMask, PMask]
                    for i in range(2):
                        hd95 = bratsUtils.getHd95(results[i], masks[i])
                        #ignore edgcases in which no distance could be calculated
                        if (hd95 >= 0):
                            lists[i].append(hd95)'''

        #calculate mean dice scores
        meanDiceA = np.mean(diceA)
        meanDiceP = np.mean(diceP)
        meanDice = np.mean([meanDiceA, meanDiceP])
        if (meanDice >= self.bestMeanDice):
            self.bestMeanDice = meanDice
            self.bestMeanDiceEpoch = epoch
            print("best so far")
            self.saveToDiskbest_(epoch)

        #update moving avg
        self._updateMovingAvg(meanDice, epoch)

        #print metrics
        print("------ Validation epoch {} ------".format(epoch))
        print("Dice        A: {:.4f} P: {:.4f}  Mean: {:.4f} MovingAvg: {:.4f}".format(meanDiceA, meanDiceP, meanDice, self.movingAvg))
        print("Sensitivity A: {:.4f} P: {:.4f} ".format(np.mean(sensA), np.mean(sensP)))
        print("Specificity A: {:.4f} P: {:.4f} ".format(np.mean(specA), np.mean(specP)))
        self.vallog.write("{}\t{:.4f}\t{:.4f}\t{:.4f}\t {:.4f}\n".format(epoch,meanDiceA, meanDiceP,  meanDice, self.movingAvg))
        '''if logHausdorff:
            print("Hausdorff   A: {:6.2f} P: {:6.2f}\n ".format(np.mean(hdA), np.mean(hdP), ))
            self.log.write("Hausdorff   A: {:6.2f} P: {:6.2f} ".format(np.mean(hdA), np.mean(hdP)))'''

        #log metrics
        if self.experiment is not None:
            self.experiment.log_metrics({"A": meanDiceA, "P": meanDiceP, "mean": meanDice, "movingAvg": self.movingAvg}, "dice", epoch)
            self.experiment.log_metrics({"A": np.mean(sensA), "P": np.mean(sensP)}, "sensitivity", epoch)
            self.experiment.log_metrics({"A": np.mean(specA), "P": np.mean(specP)}, "specificity", epoch)
            if logHausdorff:
                self.experiment.log_metrics({"A": np.mean(hdA), "P:": np.mean(hdP)}, "hausdorff", epoch)

        #print(buckets)

        #log validation time
        if expConfig.LOG_VALIDATION_TIME:
            print("Time for validation: {:.2f}s".format(time.time() - startTime))
        print("--------------------------------")


    def logMemoryUsage(self, additionalString=""):
        if torch.cuda.is_available():
            print(additionalString + "Memory {:.0f}Mb max, {:.0f}Mb current".format(
                torch.cuda.max_memory_allocated() / 1024 / 1024, torch.cuda.memory_allocated() / 1024 / 1024))


    def saveToDisk(self, epoch):

        #gather things to save
        saveDict = {"net_state_dict": self.expConfig.net.state_dict(),
                    "optimizer_state_dict": self.expConfig.optimizer.state_dict(),
                    "epoch": epoch,
                    "bestMeanDice": self.bestMeanDice,
                    "bestMeanDiceEpoch": self.bestMeanDiceEpoch,
                    "movingAvg": self.movingAvg,
                    "bestMovingAvgEpoch": self.bestMovingAvgEpoch,
                    "bestMovingAvg": self.bestMovingAvg}
        if hasattr(self.expConfig, "lr_sheudler"):
            saveDict["lr_sheudler_state_dict"] = self.expConfig.lr_sheudler.state_dict()

        #save dict
        basePath = self.checkpointsBasePathSave + "{}".format(self.expConfig.id)
        path = basePath + "/e_{}.pt".format(epoch)
        if not os.path.exists(basePath):
            os.makedirs(basePath)
        torch.save(saveDict, path)
    def saveToDiskbest(self, epoch):

        #gather things to save
        saveDict = {"net_state_dict": self.expConfig.net.state_dict(),
                    "optimizer_state_dict": self.expConfig.optimizer.state_dict(),
                    "epoch": epoch,
                    "bestMeanDice": self.bestMeanDice,
                    "bestMeanDiceEpoch": self.bestMeanDiceEpoch,
                    "movingAvg": self.movingAvg,
                    "bestMovingAvgEpoch": self.bestMovingAvgEpoch,
                    "bestMovingAvg": self.bestMovingAvg}
        if hasattr(self.expConfig, "lr_sheudler"):
            saveDict["lr_sheudler_state_dict"] = self.expConfig.lr_sheudler.state_dict()

        #save dict
        basePath = self.checkpointsBasePathSave + "{}".format(self.expConfig.id)
        path = basePath + "/e_best.pt".format(epoch)
        if not os.path.exists(basePath):
            os.makedirs(basePath)
        torch.save(saveDict, path)
    def saveToDiskbest_(self, epoch):

        #gather things to save
        saveDict = {"net_state_dict": self.expConfig.net.state_dict(),
                    "optimizer_state_dict": self.expConfig.optimizer.state_dict(),
                    "epoch": epoch,
                    "bestMeanDice": self.bestMeanDice,
                    "bestMeanDiceEpoch": self.bestMeanDiceEpoch,
                    "movingAvg": self.movingAvg,
                    "bestMovingAvgEpoch": self.bestMovingAvgEpoch,
                    "bestMovingAvg": self.bestMovingAvg}
        if hasattr(self.expConfig, "lr_sheudler"):
            saveDict["lr_sheudler_state_dict"] = self.expConfig.lr_sheudler.state_dict()

        #save dict
        basePath = self.checkpointsBasePathSave + "{}".format(self.expConfig.id)
        path = basePath + "/e_best1.pt".format(epoch)
        if not os.path.exists(basePath):
            os.makedirs(basePath)
        torch.save(saveDict, path)

    def loadFromDisk(self, id, epoch):
        path = self._gePheckpointPathLoad(id, epoch)
        checkpoint = torch.load(path)
        self.expConfig.net.load_state_dict(checkpoint["net_state_dict"])

        #load optimizer: hack necessary because load_state_dict has bugs (See https://github.com/pytorch/pytorch/issues/2830#issuecomment-336194949)
        self.expConfig.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in self.expConfig.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    if torch.cuda.is_available():
                        state[k] = v.cuda()
                    else:
                        state[k] = v

        if "lr_sheudler_state_dict" in checkpoint:
            self.expConfig.lr_sheudler.load_state_dict(checkpoint["lr_sheudler_state_dict"])
            #Hack lr sheudle
            #self.expConfig.lr_sheudler.milestones = [250, 400, 550]

        #load best epoch score (if available)
        if "bestMeanDice" in checkpoint:
            self.bestMeanDice = checkpoint["bestMeanDice"]
            self.bestMeanDiceEpoch = checkpoint["bestMeanDiceEpoch"]

        #load moving avg if available
        if "movingAvg" in checkpoint:
            self.movingAvg = checkpoint["movingAvg"]

        #load best moving avg epoch if available
        if "bestMovingAvgEpoch" in checkpoint:
            self.bestMovingAvgEpoch = checkpoint["bestMovingAvgEpoch"]
        if "bestMovingAvg" in checkpoint:
            self.bestMovingAvg = checkpoint["bestMovingAvg"]

        return checkpoint["epoch"]

    def _gePheckpointPathLoad(self, id, epoch):
        return self.checkpointsBasePathLoad + "{}/e_{}.pt".format(id, epoch)

    def _updateMovingAvg(self, validationMean, epoch):
        if self.movingAvg == 0:
            self.movingAvg = validationMean
        else:
            alpha = self.EXPONENTIAL_MOVING_AVG_ALPHA
            self.movingAvg = self.movingAvg * alpha + validationMean * (1 - alpha)

        if self.bestMovingAvg < self.movingAvg:
            self.bestMovingAvg = self.movingAvg
            self.bestMovingAvgEpoch = epoch
            self.saveToDiskbest(epoch)
