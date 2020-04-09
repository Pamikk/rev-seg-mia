from comet_ml import Experiment, ExistingExperiment
import sys
sys.path.append("..")
import torch
import torch.optim as optim
import torch.nn as nn
import bratsUtils
import torch.nn.functional as F
import random

id = random.getrandbits(64)

#restore experiment
#VALIDATE_ALL = False
#PREDICT = True
#RESTORE_ID = 13208142648257160921
#RESTORE_EPOCH = "best"
#LOG_COMETML_EXISTING_EXPERIMENT = ""

#general settings
Depth = 1
SAVE_CHECKPOINTS = True #set to true to create a checkpoint at every epoch
EXPERIMENT_TAGS = ["bugfreeFinalDrop"]
EXPERIMENT_NAME = "Baseline"+str(Depth)
EPOCHS = 1000
BATCH_SIZE = 1
VIRTUAL_BATCHSIZE = 1
VALIDATE_EVERY_K_EPOCHS = 1
SAVE_EVERY_K_EPOCHS = 25
INPLACE = True

#hyperparameters
#CHANNELS = [80,160,320,640]
#CHANNELS = [96,192,384,768]
CHANNELS =[60,120,240,480]
INITIAL_LR = 1e-4
L2_REGULARIZER = 1e-5

#logging settings
LOG_EVERY_K_ITERATIONS = 50 #0 to disable logging
LOG_MEMORY_EVERY_K_ITERATIONS = False
LOG_MEMORY_EVERY_EPOCH = True
LOG_EPOCH_TIME = True
LOG_VALIDATION_TIME = True
LOG_HAUSDORFF_EVERY_K_EPOCHS = 0 #must be a multiple of VALIDATE_EVERY_K_EPOCHS
LOG_COMETML = False
LOG_PARAMCOUNT = True
LOG_LR_EVERY_EPOCH = True

#data and augmentation
TRAIN_ORIGINAL_CLASSES = False #train on original 5 classes
DATASET_WORKERS = 1
SOFT_AUGMENTATION = False #Soft augmetation directly works on the 3 classes. Hard augmentation augments on the 5 orignal labels, then takes the argmax
NN_AUGMENTATION = True #Has priority over soft/hard augmentation. Uses nearest-neighbor interpolation
DO_ROTATE = True
DO_SCALE = True
DO_FLIP = True
DO_ELASTIC_AUG = False
DO_INTENSITY_SHIFT = True
#RANDOM_CROP = [128, 128, 128]

ROT_DEGREES = 20
SCALE_FACTOR = 1.1
SIGMA = 10
MAX_INTENSITY_SHIFT = 0.1

if LOG_COMETML:
    if not "LOG_COMETML_EXISTING_EXPERIMENT" in locals():
        experiment = Experiment(api_key="", project_name="", workspace="")
    else:
        experiment = ExistingExperiment(api_key="", previous_experiment=LOG_COMETML_EXISTING_EXPERIMENT, project_name="", workspace="")
else:
    experiment = None

#network funcitons
if TRAIN_ORIGINAL_CLASSES:
    loss = bratsUtils.bratsDiceLossOriginal5
else:
    #loss = bratsUtils.bratsDiceLoss
    def loss(outputs, labels):
        return bratsUtils.bratsDiceLoss(outputs, labels, nonSquared=True)

class ResidualInner(nn.Module):
    def __init__(self, channels, groups):
        super(ResidualInner, self).__init__()
        self.gn = nn.GroupNorm(groups, channels)
        self.conv = nn.Conv3d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv(F.leaky_relu(self.gn(x), inplace=INPLACE))
        return x

def makeSequence(channels):
    groups = CHANNELS[0] // 4
    fBlock = ResidualInner(channels, groups)
    gBlock = ResidualInner(channels, groups)
    return nn.Sequential(*[fBlock,gBlock])
def makeComponent(channels,blockCount):
    modules = []
    for i in range(blockCount):
        modules.append(makeSequence(channels))
    return nn.Sequential(*modules)
class EncoderModule(nn.Module):
    def __init__(self, inChannels, outChannels, depth, downsample=True):
        super(EncoderModule, self).__init__()
        self.downsample = downsample
        if downsample:
            self.conv = nn.Conv3d(inChannels, outChannels, 1)
        self.Blocks = makeComponent(outChannels, depth)

    def forward(self, x):
        if self.downsample:
            x = F.avg_pool3d(x, 2)
            x = self.conv(x) #increase number of channels
        x = self.Blocks(x)
        return x
class DecoderModule(nn.Module):
    def __init__(self, inChannels, outChannels, depth, upsample=True):
        super(DecoderModule, self).__init__()
        self.Blocks = makeComponent(inChannels, depth)
        self.upsample = upsample
        if self.upsample:
            self.conv = nn.Conv3d(inChannels, outChannels, 1)

    def forward(self, x):
        x = self.Blocks(x)
        if self.upsample:
            x = self.conv(x)
            x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        return x
def getChannelsAtIndex(index):
    if index < 0: index = 0
    if index >= len(CHANNELS): index = len(CHANNELS) - 1
    return CHANNELS[index]//2
class NoNewNet(nn.Module):
    def __init__(self):
        super(NoNewNet, self).__init__()
        channels = CHANNELS[0]//2
        ecdepth = Depth
        dcdepth = 1
        self.levels = 5
        self.firstConv = nn.Conv3d(1,channels, 3, padding=1, bias=False)
        self.lastConv = nn.Conv3d(channels, 2, 1, bias=True)
        #create encoder levels
        encoderModules = []
        for i in range(self.levels):
            encoderModules.append(EncoderModule(getChannelsAtIndex(i - 1), getChannelsAtIndex(i), ecdepth, i != 0))
        self.encoders = nn.ModuleList(encoderModules)

        #create decoder levels
        decoderModules = []
        for i in range(self.levels):
            decoderModules.append(DecoderModule(getChannelsAtIndex(self.levels - i - 1), getChannelsAtIndex(self.levels - i - 2), dcdepth, i != (self.levels -1)))
        self.decoders = nn.ModuleList(decoderModules)

    def forward(self, x):
        x = self.firstConv(x)
        inputStack = []
        for i in range(self.levels):
            x = self.encoders[i](x)
            if i < self.levels - 1:
                inputStack.append(x)

        for i in range(self.levels):
            x = self.decoders[i](x)
            if i < self.levels - 1:
                x = x + inputStack.pop()

        x = self.lastConv(x)
        x = torch.sigmoid(x)
        return x

net = NoNewNet()

optimizer = optim.Adam(net.parameters(), lr=INITIAL_LR, weight_decay=L2_REGULARIZER)
lr_sheudler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max', factor=0.5, patience=15)