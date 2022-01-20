from thop import profile
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class self_attention(nn.Module):
    def __init__(self, input):
        super(self_attention, self).__init__()
        self.in_channels = input
        self.query = nn.Conv2d(input, input // 8, kernel_size=1, stride=1)
        self.key = nn.Conv2d(input, input // 8, kernel_size=1, stride=1)
        self.value = nn.Conv2d(input, input, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x,y):
        batch_size, channels, height, width = x.shape
        q = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        k = self.key(y).view(batch_size, -1, height * width)
        v = self.value(x).view(batch_size, -1, height * width)
        attn_matrix = torch.bmm(q, k)
        attn_matrix = 1-self.softmax(attn_matrix)
        #print(attn_matrix.shape,v.shape)
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))
        out = out.view(*x.shape)

        return self.gamma * out + x

class DenseUnet(nn.Module):
    def __init__(self, n_class):
        super(DenseUnet, self).__init__()
        self.num_layers = 161
        if self.num_layers == 169:
            Thermal_raw_model = models.densenet169(pretrained=True)
            RGB_raw_model = models.densenet169(pretrained=True)
            self.bt0 = 64
            self.bt1 = 128
            self.bt2= 256
            self.bt3= 640
            self.bt4 = 832
        elif self.num_layers == 121:
            Thermal_raw_model = models.densenet121(pretrained=True)
            RGB_raw_model = models.densenet121(pretrained=True)
            self.bt0 = 64
            self.bt1 = 128
            self.bt2= 256
            self.bt3= 512
            self.bt4 = 512
        elif self.num_layers == 201:
            Thermal_raw_model = models.densenet201(pretrained=True)
            RGB_raw_model = models.densenet201(pretrained=True)
            self.bt0 = 64
            self.bt1 = 128
            self.bt2 = 256
            self.bt3 = 896
            self.bt4 = 960
        elif self.num_layers == 161:
            Thermal_raw_model = models.densenet161(pretrained=True)
            RGB_raw_model = models.densenet161(pretrained=True)
            self.bt_initial = 96
            self.bt0 = 96
            self.bt1 = 192
            self.bt2 = 384
            self.bt3 = 1056
            self.bt4 = 1104
        ########  Thermal ENCODER  ########
        self.TE_conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)   #Initial_block----thermal_image_encoder_conv0  (DenseNet)
        self.TE_conv0.weight.data = torch.unsqueeze(torch.mean(Thermal_raw_model.features.conv0.weight.data, dim=1), dim=1)
        self.TE_bn0= Thermal_raw_model.features.norm0
        self.TE_relu0 = Thermal_raw_model.features.relu0
        self.TE_pool0 = Thermal_raw_model.features.pool0   #out: H/4 * W/4

        self.TE_d1 = Thermal_raw_model.features.denseblock1  #thermal_image_encoder_denseblock1  (DenseNet) H/4 * W/4
        self.TE_t1 = Thermal_raw_model.features.transition1  #thermal_image_encoder_transition1  (DenseNet) H/8 * W/8

        self.TE_d2 = Thermal_raw_model.features.denseblock2  #out:H/8 * W/8
        self.TE_t2 = Thermal_raw_model.features.transition2  #out:H/16 * W/16

        self.TE_d3 = Thermal_raw_model.features.denseblock3  #out:H/16 * W/16
        self.TE_t3 = Thermal_raw_model.features.transition3  #out:H/32 * W/32

        self.TE_d4 = Thermal_raw_model.features.denseblock4  #out:H/32 * W/32
        self.TE_t4 = nn.Sequential(                         # out:H/64 * W/64
            nn.BatchNorm2d(2*self.bt4),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*self.bt4, self.bt4, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
        self.TE_att4 = self_attention(input=self.bt4)
        ########  RGB ENCODER  ########
        self.RE_conv0 = RGB_raw_model.features.conv0  #Initial_block----RGB_image_encoder_conv0  (DenseNet)
        self.RE_bn0 = RGB_raw_model.features.norm0
        self.RE_relu0 = RGB_raw_model.features.relu0
        self.RE_pool0 = RGB_raw_model.features.pool0

        self.RE_d1 = RGB_raw_model.features.denseblock1
        self.RE_t1 = RGB_raw_model.features.transition1

        self.RE_d2 = RGB_raw_model.features.denseblock2
        self.RE_t2 = RGB_raw_model.features.transition2

        self.RE_d3 = RGB_raw_model.features.denseblock3
        self.RE_t3 = RGB_raw_model.features.transition3

        self.RE_d4 = RGB_raw_model.features.denseblock4
        self.RE_t4 = nn.Sequential(
            nn.BatchNorm2d(2*self.bt4),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*self.bt4, self.bt4, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
        self.RE_att4 = self_attention(input=self.bt4)
        ########  DECODER  ########
        self.feature_up4=Up_decoder(in_f=self.bt4, out_f=self.bt3,op=True)
        self.feature_decoder4= Feature_decoder(in_feature=self.bt3, out_feature=self.bt3)

        self.feature_up3=Up_decoder(in_f=self.bt3, out_f=self.bt2)
        self.feature_decoder3 = Feature_decoder(in_feature=self.bt2, out_feature=self.bt2)

        self.feature_up2=Up_decoder(in_f=self.bt2, out_f=self.bt1)
        self.feature_decoder2 = Feature_decoder(in_feature=self.bt1, out_feature=self.bt1)

        self.feature_up1=Up_decoder(in_f=self.bt1, out_f=self.bt0)
        self.feature_decoder1 = Feature_decoder(in_feature=self.bt0, out_feature=self.bt0)

        self.feature_up0=Up_decoder(in_f=self.bt0, out_f=self.bt_initial)
        self.feature_decoder0 = Feature_decoder(in_feature=self.bt_initial, out_feature=self.bt_initial)
        self.classification = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channels=self.bt_initial, out_channels=n_class, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
        )

    def forward(self, input):
        rgb = input[:, :3]
        thermal = input[:, 3:]
        verbose = False
        # encoder
        if verbose: print("encoder:",'\n',"rgb.size() original: ", rgb.size())  # (480, 640)
        ######################################################################
        rgb = self.RE_conv0(rgb)
        if verbose: print("rgb.size() after conv07*7: ", rgb.size())  # (240, 320)
        rgb = self.RE_bn0(rgb)
        rgb = self.RE_relu0(rgb)
        thermal = self.TE_conv0(thermal)
        thermal = self.TE_bn0(thermal)
        thermal = self.TE_relu0(thermal)
        fuse_initial = rgb + thermal


        rgb0 = self.RE_pool0(fuse_initial)
        if verbose: print("rgb.size() after pool: ", rgb0.size())  # (120, 160)
        thermal0 = self.TE_pool0(thermal)
        fuse0=rgb0+thermal0
        ######################################################################
        rgb1_ = self.RE_d1(fuse0)
        rgb1 = self.RE_t1(rgb1_)
        if verbose: print("rgb.size() after densely1 & transition1: ", rgb1_.size(),rgb1.size())  # (120, 160)
        thermal1_ = self.TE_d1(thermal0)
        thermal1 = self.TE_t1(thermal1_)
        fuse1 = rgb1 + thermal1
        ######################################################################
        rgb2_ = self.RE_d2(fuse1)
        rgb2 = self.RE_t2(rgb2_)
        if verbose: print("rgb.size() after densely2 & transition2: ", rgb2_.size(),rgb2.size())  # (60, 80)
        thermal2_ = self.TE_d2(thermal1)
        thermal2 = self.TE_t2(thermal2_)
        fuse2 = rgb2 + thermal2
        ######################################################################
        rgb3_ = self.RE_d3(fuse2)
        rgb3 = self.RE_t3(rgb3_)
        if verbose: print("rgb.size() after densely3 & transition3: ", rgb3_.size(),rgb3.size())  # (30, 40)
        thermal3_ = self.TE_d3(thermal2)
        thermal3 = self.TE_t3(thermal3_)
        fuse3 = rgb3 + thermal3
        ######################################################################
        rgb4_ = self.RE_d4(fuse3)
        rgb4 = self.RE_t4(rgb4_)
        if verbose: print("rgb.size() after densely4 & transition4: ", rgb4_.size(),rgb4.size())  # (15, 20)
        thermal4_ = self.TE_d4(thermal3)
        thermal4 = self.TE_t4(thermal4_)
        rgb4 = self.RE_att4(rgb4, thermal4)
        thermal4 = self.TE_att4(thermal4, rgb4)
        fuse4 = rgb4 + thermal4
        ######################################################################
        # decoder
        fuse_up4 = self.feature_up4(fuse4)
        fuse_d4 = self.feature_decoder4(fuse_up4)

        fuse_up3 = self.feature_up3(fuse_d4)
        fuse_d3 = self.feature_decoder3(fuse_up3)

        fuse_up2 = self.feature_up2(fuse_d3)
        fuse_d2 = self.feature_decoder2(fuse_up2)

        fuse_up1 = self.feature_up1(fuse_d2)
        fuse_d1 = self.feature_decoder1(fuse_up1)

        fuse_up0 = self.feature_up0(fuse_d1)
        fuse_d0 = self.feature_decoder0(fuse_up0)

        fuse = self.classification(fuse_d0)
        return fuse

class Up_decoder(nn.Module):
    def __init__(self, in_f, out_f,k=2, s=2, p=0, b=False, op=False):
        super(Up_decoder, self).__init__()
        self.pad=False
        self.conv1 = nn.Conv2d(in_f, out_f, kernel_size=3,stride=1, padding=1,bias=b)
        self.bn1 = nn.BatchNorm2d(out_f)
        self.up=nn.ConvTranspose2d(out_f, out_f, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_f)
        self.relu = nn.ReLU(inplace=True)
        self.residual=nn.ConvTranspose2d(in_f, out_f, kernel_size=k, stride=s, padding=p, bias=False)
        if op:
            self.pad=True
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.up(out)
        out = self.bn(out)
        if self.pad:
            out=F.pad(out, pad=[0, 0, 1, 0], mode="replicate")
            residual=F.pad(residual, pad=[0, 0, 1, 0], mode="replicate")
        out += residual
        out = self.relu(out)

        return out

class Feature_decoder(nn.Module):
    def __init__(self, in_feature, out_feature,k=3, s=1, p=1, b=False):
        super(Feature_decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_feature, out_feature, kernel_size=k, stride=s, padding=p, bias=b)
        self.bn1 = nn.BatchNorm2d(out_feature)
        self.conv2 = nn.Conv2d(out_feature, out_feature,kernel_size=k, stride=s, padding=p, bias=b)
        self.bn2 = nn.BatchNorm2d(out_feature)
        self.relu = nn.ReLU(inplace=True)
        self.residual = nn.Conv2d(in_feature, out_feature, kernel_size=1, stride=1, padding=0, bias=b)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        residual =self.residual(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


########################## Info_NestSeg ################
def unit_test():
    num_minibatch = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rgb = torch.randn(num_minibatch, 3, 480, 640).cuda(0)
    thermal = torch.randn(num_minibatch, 1, 480, 640).cuda(0)
    #print(DenseUnet(9))
    model =  DenseUnet(9).to(device)
    input = torch.cat((rgb, thermal), dim=1)
    macs, params = profile(model, inputs=(input, ))
    print(macs, params)

if __name__ == '__main__':
    unit_test()

