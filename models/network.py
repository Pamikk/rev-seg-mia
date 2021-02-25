import torch.nn as nn
import torch.nn.functional as F

from .backbone import ResNet,conv1x1,conv3x3 
__all__=['Networkv2','Networkv3','Networkv4','YOLO','YOLOu']
def NetAPI(cfg,net):
    networks = {'yolo':YOLO}
    return networks[net](cfg)
class BaseBlock(nn.Module):
    multiple=2
    def __init__(self,in_channels,channels,stride=1):
        super(BaseBlock,self).__init__()
        self.conv1 = conv1x1(in_channels,channels,stride)
        self.relu = nn.LeakyReLU(0.1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = conv3x3(channels,channels*BaseBlock.multiple)
        if in_channels != channels*BaseBlock.multiple or stride!=1:
            self.downsample =  nn.Sequential(conv1x1(in_channels,channels*BaseBlock.multiple,stride),
                                            nn.BatchNorm2d(channels*BaseBlock.multiple))
        else:
            self.downsample = nn.Identity()
        self.bn2 = nn.BatchNorm2d(channels*BaseBlock.multiple)
    def forward(self,x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)

        if self.downsample != None:
            x = self.downsample(x)
        y += x
        y = self.relu(y)

        return y
class Bottleneck(nn.Module):
    multiple = 2
    def __init__(self,in_channels,channels,stride=1):
        super(Bottleneck,self).__init__()
        self.conv1 = conv1x1(in_channels,channels,stride=stride)
        self.relu = nn.LeakyReLU(0.01)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = conv3x3(channels,channels)
        if in_channels != channels*Bottleneck.multiple or stride!=1:
            self.downsample =  nn.Sequential(conv1x1(in_channels,channels*Bottleneck.multiple,stride),
                                            nn.BatchNorm2d(channels*Bottleneck.multiple))
        else:
            self.downsample = nn.Identity()
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = conv1x1(channels,Bottleneck.multiple*channels)
        self.bn3 = nn.BatchNorm2d(Bottleneck.multiple*channels)
    def forward(self,x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)

        y = self.conv3(y)
        y = self.bn3(y)
        y = self.relu(y)

        x = self.downsample(x)
        y += x
        y = self.relu(y)

        return y
class Networkv2(nn.Module):
    def __init__(self,cfg,pretrained=True):
        super(Networkv2,self).__init__()
        self.feat = ResNet(cfg.res)
        self.channel = 256
        if pretrained:
            self.feat.load_from_url()
        self.pred = self.make_prediction(self.feat.channels[0],cls_num,Bottleneck)
        
    def make_prediction(self,in_channel,out_channel,block):
        blocks = [block(in_channel,self.channel)]
        blocks.append(nn.Conv2d(self.channel*block.multiple,out_channel,kernel_size=3,padding=1,stride=1,bias=False))
        blocks.append(nn.BatchNorm2d(out_channel))
        return nn.Sequential(*blocks)

    def forward(self,x):
        feats = self.feat(x)
        feats = feats[0]
        return self.pred(feats)
class Networkv3(nn.Module):
    def __init__(self,res,int_shape,cls_num,pretrained=True):
        super(Networkv3,self).__init__()
        self.feat = ResNet(res)
        self.channel = self.feat.channels[-1]
        channels = self.feat.channels
        if pretrained:
            self.feat.load_from_url()
        self.pred = self.make_prediction(cls_num,Bottleneck)
        layers = []
        self.decode_depth = 1
        self.levels = 4
        for i in range(self.levels):
            if i< self.levels-1:
                layers.append(self.make_layers(channels[i],channels[i+1],Bottleneck,self.decode_depth))
            else:
                layers.append(self.make_layers(channels[i],self.channel,Bottleneck,self.decode_depth))
        self.decoders = nn.ModuleList(layers)
        
    def make_layers(self,in_channel,channel,block,depth=1):
        blocks = [block(in_channel,channel//block.multiple)]
        if depth > 1:
            blocks += [block(channel,channel//block.multiple) for _ in range(1,depth)]
        return nn.Sequential(*blocks)        
    def make_prediction(self,out_channel,block):
        #32x32 for 256x256 as input
        blocks = [block(self.channel,self.channel,stride=2)]
        blocks.append(nn.Conv2d(self.channel*block.multiple,out_channel,kernel_size=3,padding=1,stride=1,bias=False))
        blocks.append(nn.BatchNorm2d(out_channel))
        return nn.Sequential(*blocks)

    def forward(self,x):
        feats = self.feat(x)
        x = feats[0]
        for i in range(self.levels):
            x = self.decoders[i](x)
            if i < self.levels-1:
                x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False) + feats[i+1]      
        res = self.pred(x) 
        return [res]
class Networkv4(nn.Module):
    def __init__(self,res,int_shape,cls_num,pretrained=True):
        super(Networkv4,self).__init__()
        self.feat = ResNet(res)
        self.channel = self.feat.channels[-1]
        channels = self.feat.channels
        if pretrained:
            self.feat.load_from_url()
        preds = []
        layers = []
        self.decode_depth = 1
        self.levels = 4
        for i in range(self.levels):
            if i< self.levels-1:
                layers.append(self.make_layers(channels[i],channels[i+1],Bottleneck,self.decode_depth))
                preds.append(self.make_prediction(channels[i+1],cls_num,Bottleneck))
            else:
                layers.append(self.make_layers(channels[i],self.channel,Bottleneck,2))
                preds.append(self.make_prediction(self.channel,cls_num,Bottleneck))
            
        self.decoders = nn.ModuleList(layers)
        self.pred = nn.ModuleList(preds)
        
    def make_layers(self,in_channel,channel,block,depth=1):
        blocks = [block(in_channel,channel//block.multiple)]
        if depth > 1:
            blocks += [block(channel,channel//block.multiple) for _ in range(1,depth)]
        return nn.Sequential(*blocks)        
    def make_prediction(self,in_channel,out_channel,block):
        #32x32 for 256x256 as input
        blocks = [block(in_channel,self.channel)]
        blocks.append(nn.Conv2d(self.channel*block.multiple,out_channel,kernel_size=3,padding=1,stride=1,bias=False))
        blocks.append(nn.BatchNorm2d(out_channel))
        return nn.Sequential(*blocks)

    def forward(self,x):
        feats = self.feat(x)
        x = feats[0]
        outs = list(range(self.levels))
        for i in range(self.levels):
            x = self.decoders[i](x)
            outs[i] = self.pred[i](x)
            if i < self.levels-1:
                x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False) + feats[i+1]      
        return [outs[-1]]
class YOLO(nn.Module):
    def __init__(self,cfg):
        super(YOLO,self).__init__()
        self.depths = [1,2,8,8,4]
        self.levels = len(self.depths)
        channels = [32,64,128,256,512]
        self.relu = nn.LeakyReLU(0.1)
        self.block1 = nn.Sequential(conv3x3(3,32),nn.BatchNorm2d(32),self.relu)
        self.block2 = nn.Sequential(conv3x3(32,64,stride=2),nn.BatchNorm2d(64),self.relu)
        self.in_channel = 64
        self.in_channels = []
        encoders = []
        for i in range(self.levels-1):
            encoders.append(self.make_encoders(channels[i],BaseBlock,depth=self.depths[i],downsample=True))
        encoders.append(self.make_encoders(channels[-1],BaseBlock,depth=self.depths[-1],downsample=False))
        self.encoders = nn.ModuleList(encoders)
        self.channel = self.in_channels[-1]
        self.pred = self.make_prediction(cfg.anchor_num*(5+cfg.cls_num))
        #for i in range(self.levels)
    def make_encoders(self,channel,block,depth=1,downsample=False):
        blocks = [block(self.in_channel,channel)]
        if depth > 1:
            blocks += [block(channel*block.multiple,channel) for _ in range(1,depth)]
        if downsample:
            out_channel = channel*block.multiple*2
            blocks.append(nn.Sequential(conv3x3(channel*block.multiple,out_channel,stride=2),nn.BatchNorm2d(out_channel),self.relu))
        else:
            out_channel = channel*block.multiple
        self.in_channel = out_channel
        self.in_channels.insert(0,self.in_channel)
        return nn.Sequential(*blocks)
    def make_prediction(self,out_channel,stride=1):
        #32x32 for 256x256 as input
        return nn.Sequential(conv3x3(self.in_channel,out_channel,stride),nn.BatchNorm2d(out_channel))
    def make_decoders(self,channel,block,depth=1):
        pass
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)

        x1 = self.encoders[0](x)
        
        x2 = self.encoders[1](x1)
        
        x3 = self.encoders[2](x2)

        x4 = self.encoders[3](x3)

        x5 = self.encoders[4](x4)

        return self.pred(x5)
class YOLOu(nn.Module):
    def __init__(self,res,int_cls_num,cls_num):
        super(YOLOu,self).__init__()
        self.depths = [1,2,8,8,4]
        self.levels = len(self.depths)
        channels = [32,64,128,256,512]
        self.relu = nn.LeakyReLU(0.1)
        self.block1 = nn.Sequential(conv3x3(3,32),nn.BatchNorm2d(32),self.relu)
        self.block2 = nn.Sequential(conv3x3(32,64,stride=2),nn.BatchNorm2d(64),self.relu)
        self.in_channel = 64
        self.in_channels = [64]
        encoders = []
        for i in range(self.levels-1):
            encoders.append(self.make_encoders(channels[i],BaseBlock,depth=self.depths[i],downsample=True))
        encoders.append(self.make_encoders(channels[-1],BaseBlock,depth=self.depths[-1],downsample=False))
        self.encoders = nn.ModuleList(encoders)
        self.channel = self.in_channels[-1]
        self.pred1 = self.make_prediction(5)
        decoders = []
        for i in range(1,self.levels-1):
            decoders.append(BaseBlock(self.in_channel,self.in_channels[i+1]//BaseBlock.multiple))
            self.in_channel = self.in_channels[i+1]
        decoders.append(BaseBlock(self.in_channel,self.in_channels[i+1]//BaseBlock.multiple,stride=2))
        self.in_channel = self.in_channels[i+1]
        self.decoders = nn.ModuleList(decoders)
        self.pred2 = self.make_prediction(cls_num,stride=2)
    def make_encoders(self,channel,block,depth=1,downsample=False):
        blocks = [block(self.in_channel,channel)]
        if depth > 1:
            blocks += [block(channel*block.multiple,channel) for _ in range(1,depth)]
        if downsample:
            out_channel = channel*block.multiple*2
            blocks.append(nn.Sequential(conv3x3(channel*block.multiple,out_channel,stride=2),nn.BatchNorm2d(out_channel),self.relu))
        else:
            out_channel = channel*block.multiple
        self.in_channel = out_channel
        self.in_channels.insert(0,self.in_channel)
        return nn.Sequential(*blocks)
    def make_prediction(self,out_channel,stride=1):
        return nn.Sequential(conv3x3(self.in_channel,out_channel,stride),nn.BatchNorm2d(out_channel))
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)

        x1 = self.encoders[0](x)
        
        x2 = self.encoders[1](x1)
        
        x3 = self.encoders[2](x2)

        x4 = self.encoders[3](x3)

        x5 = self.encoders[4](x4)

        x = x4+x5

        x = self.decoders[0](x)
        x = F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False) + x3

        x = self.decoders[1](x)
        x = F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False) + x2

        x = self.decoders[2](x)
        x = F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False) + x1

        x = self.decoders[3](x)
        return [self.pred1(x5),self.pred2(x)]


    