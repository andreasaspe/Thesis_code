import torch
import torch.nn as nn


#Helping function
def DoubleConv(in_channels, feature_maps, dropout):
        double_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels,out_channels=feature_maps,kernel_size=3, stride = 1, padding = 1),
            nn.BatchNorm3d(feature_maps),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_channels=feature_maps,out_channels=feature_maps,kernel_size=3, stride = 1, padding = 1),
            nn.BatchNorm3d(feature_maps),
            nn.LeakyReLU(inplace=True),
        )
        return double_conv


class LocalAppearance(nn.Module):
    def __init__(self, dropout):
        super(LocalAppearance, self).__init__()

        #Double convolutions
        self.conv_down1 = DoubleConv(1,64,dropout)
        self.conv_down2 = DoubleConv(64,64,dropout)
        self.conv_down3 = DoubleConv(64,64,dropout)
        self.conv_down4 = DoubleConv(64,64,dropout)
        self.bottom = DoubleConv(64,64,dropout)
        self.conv_up4 = DoubleConv(128,64,dropout)
        self.conv_up3 = DoubleConv(128,64,dropout)
        self.conv_up2 = DoubleConv(128,64,dropout)
        self.conv_up1 = DoubleConv(128,64,dropout)
        

        #Average pooling
        self.avgpool = nn.AvgPool3d(kernel_size=2,stride=2)

        #Dropout
        self.dropout = nn.Dropout3d(p=dropout)

        #Linear upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')

        #Output
        self.output = nn.Conv3d(in_channels=64,out_channels=1,kernel_size=3, stride = 1, padding = 1)

        nn.init.normal_(self.output.weight, std=0.0001)

    def forward(self, image):
        #Contracting
        layer1_skip = self.conv_down1(image)
        x = self.avgpool(layer1_skip)
        x = self.dropout(x)

        layer2_skip = self.conv_down2(x)
        x = self.avgpool(layer2_skip)
        x = self.dropout(x)

        layer3_skip = self.conv_down3(x)
        x = self.avgpool(layer3_skip)
        x = self.dropout(x)

        layer4_skip = self.conv_down4(x)
        x = self.avgpool(layer4_skip)
        x = self.dropout(x)

        #Parallel
        x = self.bottom(x)

        #Expanding
        x = self.upsample(x)
        x = self.dropout(x)
        x = self.conv_up4(torch.cat((layer4_skip,x),dim=1))

        x = self.upsample(x)
        x = self.dropout(x)
        x = self.conv_up3(torch.cat((layer3_skip,x),dim=1))

        x = self.upsample(x)
        x = self.dropout(x)
        x = self.conv_up2(torch.cat((layer2_skip,x),dim=1))

        x = self.upsample(x)
        x = self.dropout(x)
        x = self.conv_up1(torch.cat((layer1_skip,x),dim=1))

        output = self.output(x)
        return output
    




    
#Second part of network
class SpatialConfiguration(nn.Module):
    def __init__(self):
        super(SpatialConfiguration, self).__init__()

        #Convolutional layer
        # self.four_convs = nn.Sequential(
        #     nn.Conv3d(in_channels=8,out_channels=64,kernel_size=7, stride = 1, padding = 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels=64,out_channels=64,kernel_size=7, stride = 1, padding = 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels=64,out_channels=64,kernel_size=7, stride = 1, padding = 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels=64,out_channels=8,kernel_size=7, stride = 1, padding = 1),
        #     nn.ReLU(inplace=True),
        # )

        kernel_size = (7,7,7)
        #stride = (1, 1, 1)

        # Calculate padding values
        padding = (
            (kernel_size[0] - 1) // 2,  # Depth
            (kernel_size[1] - 1) // 2,  # Height
            (kernel_size[2] - 1) // 2   # Width
        )

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=7, stride = 1, padding = padding)
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=7, stride = 1, padding = padding)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=7, stride = 1, padding = padding)
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=1, kernel_size=7, stride = 1, padding = padding)

        nn.init.normal_(self.conv4.weight, std=0.0001)

        self.activation1 = nn.LeakyReLU(inplace=True)
        self.activation2 = nn.Tanh()

        #Linear upsampling
        self.upsample = nn.Upsample(scale_factor=4, mode='trilinear')

    def forward(self,image):
        # #Do the four convolutions
        # x = self.four_convs(image)

        x = self.conv1(image)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.activation1(x)
        x = self.conv3(x)
        x = self.activation1(x)
        x = self.conv4(x)
        x = self.activation2(x)

        #Upsample to original resolution which is four times as big
        output = self.upsample(x)

        return output


#COMBINED NETWORK
class VertebraeLocalisationNet(nn.Module): #SpatialConfigurationNet
    def __init__(self,dropout):
        super(VertebraeLocalisationNet, self).__init__()

        #Define the two parts of the network
        self.LocalAppearance = LocalAppearance(dropout)
        self.SpatialConfiguration = SpatialConfiguration()

        #Average pooling
        self.avgpool = nn.AvgPool3d(kernel_size=4)


    def forward(self,image):
        #Call the Local Apperance network
        LocalAppearance_output = self.LocalAppearance(image)

        #Downsample resolution for input to Spatial Configuration
        x = self.avgpool(LocalAppearance_output)

        #Call the Spatial Configuration network
        SpatialConfiguration_output = self.SpatialConfiguration(x)

        #Do elementwise multiplication
        output = LocalAppearance_output*SpatialConfiguration_output

        return output





if __name__ == "__main__":
    image = torch.rand((1,1,96,96,128))
    model = VertebraeLocalisationNet(0.0)
    print(model)
    #Call model
    print(model(image))


