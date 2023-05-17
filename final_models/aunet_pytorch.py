import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        # number of input channels is a number of filters in the previous layer
        # number of output channels is a number of filters in the current layer
        # "same" convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x
    
# class Up(nn.Mode)


class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(
                F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True
            ),
            nn.BatchNorm2d(n_coefficients),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(
                F_l, n_coefficients, kernel_size=2, stride=2, padding=0, bias=True
            ),
            nn.BatchNorm2d(n_coefficients),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        print(psi.shape, skip_connection.shape)
        psi = F.upsample(psi, scale_factor=2, mode="bilinear")
        print(psi.shape, skip_connection.shape)

        out = skip_connection * psi
        return out


class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        psi = F.upsample(psi, scale_factor=2, mode="bilinear")

        out = skip_connection * psi
        return out


class AttentionUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, fd=[32, 64, 128, 256, 512]):
        super(AttentionUNet, self).__init__()

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(img_ch, fd[0])
        self.Conv2 = ConvBlock(fd[0], fd[1])
        self.Conv3 = ConvBlock(fd[1], fd[2])
        self.Conv4 = ConvBlock(fd[2], fd[3])
        self.Conv5 = ConvBlock(fd[3], fd[4])

        self.Att5 = AttentionBlock(F_g=fd[4], F_l=fd[3], n_coefficients=fd[3])
        self.Up5 = nn.Upsample(scale_factor=2)
        self.UpConv5 = ConvBlock(fd[4]+fd[3], fd[3])

        self.Att4 = AttentionBlock(F_g=fd[3], F_l=fd[2], n_coefficients=fd[2])
        self.Up4 = nn.Upsample(scale_factor=2)
        self.UpConv4 = ConvBlock(fd[3]+fd[2], fd[2])

        self.Att3 = AttentionBlock(F_g=fd[2], F_l=fd[1], n_coefficients=fd[1])
        self.Up3 = nn.Upsample(scale_factor=2)
        self.UpConv3 = ConvBlock(fd[2]+fd[1], fd[1])


        self.Att2 = AttentionBlock(F_g=fd[1], F_l=fd[0], n_coefficients=fd[0])
        self.Up2 = nn.Upsample(scale_factor=2)
        self.UpConv2 = ConvBlock(fd[1]+fd[0], fd[0])

        self.Conv = nn.Conv2d(fd[0], output_ch, kernel_size=1, stride=1, padding=0)

        self.output_act = nn.Sigmoid()

    def forward(self, x):
        """
        e : encoder layers
        d : decoder layers
        s : skip-connections from encoder layers to decoder layers
        """
        print(1,x.shape)

        e1 = self.Conv1(x)
        print(2,e1.shape)

        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)
        print(3, e2.shape)

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)
        print(4, e3.shape)

        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)
        print(5, e4.shape)

        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)
        print(6, e5.shape)


        a5 = self.Att5(gate=e5,skip_connection=e4)
        d5 = self.Up5(e5)
        d5 = torch.cat((a5, d5), dim=1)
        d5 = self.UpConv5(d5)

        a4 = self.Att4(gate=d5,skip_connection=e3)
        d4 = self.Up4(d5)
        d4 = torch.cat((a4, d4), dim=1)
        d4 = self.UpConv4(d4)

        a3 = self.Att3(gate=d4,skip_connection=e2)
        d3 = self.Up3(d4)
        d3 = torch.cat((a3, d3), dim=1)
        d3 = self.UpConv3(d3)

        a2 = self.Att2(gate=d3,skip_connection=e1)
        d2 = self.Up2(d3)
        d2 = torch.cat((a2, d2), dim=1)
        d2 = self.UpConv2(d2)

        out = self.Conv(d2)

        outact = self.output_act(out)

        return outact



if __name__ == "__main__":
    X = torch.rand(1, 16, 144, 144)
    attn_unet = AttentionUNet(16, 12)

    pred = attn_unet(X)
    print(pred.shape)
    # print(summary(attn_unet,input_size=(16,128,128),batch_size=4,device='cpu'))


















# class AttentionBlock(nn.Module):
#     def __init__(self, down_c, side_c, first_out_c) -> None:
#         super(AttentionBlock).__init__()
#         self.sub_sample_factor = 2
#         self.sub_sample_kernel_size = 2
#         self.donw_conv = nn.Conv2d(
#             in_channels=down_c,
#             out_channels=first_out_c,
#             kernel_size=self.sub_sample_kernel_size,
#             stride=self.sub_sample_factor,
#             padding=0,
#             bias=False,
#         )

#         self.side_c = nn.Conv2d(
#             in_channels=side_c,
#             out_channels=first_out_c,
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             bias=True,
#         )

#         self.phi = nn.Conv2d(
#             in_channels=self.inter_channels,
#             out_channels=1,
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             bias=True,
#         )

#     def forward(self, side, down):
#         new_side = self.side_c(side)
#         new_down = self.down_c(down)

#         print(new_side.shape, new_down.shape)

#         add_both = new_side + new_down
#         print(add_both.shape)

#         phi = self.phi(add_both)
#         print(phi.shape)

#         return phi