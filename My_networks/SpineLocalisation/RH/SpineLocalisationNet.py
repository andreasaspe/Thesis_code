import torch
import torch.nn as nn

    
def DoubleConv(in_channels, feature_maps):
        double_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels,out_channels=feature_maps,kernel_size=3, stride = 1, padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=feature_maps,out_channels=feature_maps,kernel_size=3, stride = 1, padding = 1),
            nn.ReLU(inplace=True),
        )
        return double_conv


class Unet3D(nn.Module):
    def __init__(self):
        super(Unet3D, self).__init__()

        #Double convolutions
        self.conv_down1 = DoubleConv(1,64)
        self.conv_down2 = DoubleConv(64,64)
        self.conv_down3 = DoubleConv(64,64)
        self.conv_down4 = DoubleConv(64,64)
        self.bottom = DoubleConv(64,64)
        self.conv_up4 = DoubleConv(128,64)
        self.conv_up3 = DoubleConv(128,64)
        self.conv_up2 = DoubleConv(128,64)
        self.conv_up1 = DoubleConv(128,64)

        #Average pooling
        self.avgpool = nn.AvgPool3d(kernel_size=2,stride=2)

        #Linear upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')

        #Output
        self.output = nn.Conv3d(in_channels=64,out_channels=1,kernel_size=3, stride = 1, padding = 1)

    def forward(self, image):
        #Contracting
        layer1_skip = self.conv_down1(image)
        x = self.avgpool(layer1_skip)

        layer2_skip = self.conv_down2(x)
        x = self.avgpool(layer2_skip)

        layer3_skip = self.conv_down3(x)
        x = self.avgpool(layer3_skip)

        layer4_skip = self.conv_down4(x)
        x = self.avgpool(layer4_skip)

        #Parallel
        x = self.bottom(x)
        x = self.bottom(x)

        #Expanding
        x = self.upsample(x)
        x = self.conv_up4(torch.cat((layer4_skip,x),dim=1))

        x = self.upsample(x)
        x = self.conv_up3(torch.cat((layer3_skip,x),dim=1))

        x = self.upsample(x)
        x = self.conv_up2(torch.cat((layer2_skip,x),dim=1))

        x = self.upsample(x)
        x = self.conv_up1(torch.cat((layer1_skip,x),dim=1))

        output = self.output(x)
        return output
    

    #def expanding(self, image):
if __name__ == "__main__":
    image = torch.rand((1,1,128,64,64))
    model = Unet3D()
    #Call model
    model(image)







# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, feature_maps):
#         super(DoubleConv,self).__init__()
#         self.double_conv = nn.Sequential(
#             nn.Conv3d(in_channels=in_channels,out_channels=feature_maps,kernel_size=3, stride = 1, padding = 1),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(in_channels=feature_maps,out_channels=feature_maps,kernel_size=3, stride = 1, padding = 1),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, image):
#         return self.double_conv(image)


# class Unet3D(nn.Module):
#     def __init__(self):
#         super(Unet3D, self).__init__()

#         self.contract = nn.ModuleList()
#         self.contract.append(DoubleConv(1,64))
#         self.contract.append(DoubleConv(64,64))

        

#     def forward(self, image):
#         x = self.contract(image)
#         print(x)
        
#         return x