import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# scale factors for image size in Generator
factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]


class WSConv2d(nn.Module):
    """Weighted-Scaled Convolution.
    Parameters
    ----------
    gain : int
        Initialization constant (he-initialization)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * kernel_size ** 2)) ** 0.5

        # bias should not be scaled
        self.bias = self.conv.bias
        self.conv.bias = None

        # Initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class WSConvTranspose2d(nn.Module):
    """Weighted-Scaled Transposed Convolution.
    Parameters
    ----------
    gain : int
        Initialization constant (he-initialization)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super(WSConvTranspose2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * kernel_size ** 2)) ** 0.5

        # bias should not be scaled
        self.bias = self.conv.bias
        self.conv.bias = None

        # Initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super(ConvBlock, self).__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.activation = nn.LeakyReLU(0.2)
        self.norm = PixelNorm()
        self.use_pixelnorm = use_pixelnorm

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.norm(x) if self.use_pixelnorm else x
        x = self.activation(self.conv2(x))
        x = self.norm(x) if self.use_pixelnorm else x
        return x


class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels=3):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            PixelNorm(),
            WSConvTranspose2d(z_dim, in_channels, 4, 1, 0),    # 1x1 -> 4x4
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )
        self.initial_rgb = WSConv2d(in_channels, img_channels, kernel_size=1, stride=1, padding=0)

        self.progress_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList([self.initial_rgb])

        for i in range(len(factors)-1):
            conv_in_channels = int(in_channels * factors[i])
            conv_out_channels = int(in_channels * factors[i+1])
            self.progress_blocks.append(ConvBlock(conv_in_channels, conv_out_channels))
            self.rgb_layers.append(WSConv2d(conv_out_channels, img_channels, kernel_size=1, stride=1, padding=0))

    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1-alpha) * upscaled)

    def forward(self, x, alpha, steps):
        """
        x : torch.Tensor
            Input tensor
        alpha : float
            Fade in value for UpScaled and Generated Image
        steps : int
            Decides to progress the image size
            i.e: steps=0 -> 4x4 | steps=1 -> 8x8 | ...
        """
        out = self.initial(x)   # 4x4

        if steps == 0:
            return self.initial_rgb(out)

        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode='nearest')
            out = self.progress_blocks[step](upscaled)

        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)
        return self.fade_in(alpha, final_upscaled, final_out)


class Discriminator(nn.Module):
    def __init__(self, in_channels, img_channels=3):
        super(Discriminator, self).__init__()
        self.progress_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList()
        self.activation = nn.LeakyReLU(0.2)

        for i in range(len(factors)-1, 0, -1):
            conv_in_channels = int(in_channels * factors[i])
            conv_out_channels = int(in_channels * factors[i-1])
            self.progress_blocks.append(ConvBlock(conv_in_channels, conv_out_channels, use_pixelnorm=False))
            self.rgb_layers.append(WSConv2d(img_channels, conv_in_channels, kernel_size=1, stride=1, padding=0))

        # for 4x4
        self.initial_rgb = WSConv2d(img_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Block of 4x4 Image
        self.final_block = nn.Sequential(
            WSConv2d(in_channels + 1, in_channels, 3, 1, 1),   # +1 to in_channels from MiniBatch std will be concatenated
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, 1, 1, 0)
        )

    def fade_in(self, alpha, downscaled, out):
        return alpha * out + (1-alpha) * downscaled

    def minibatch_std(self, x):
        batch_statistics = torch.std(x, dim=0).mean().repet(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, alpha, steps):
        curr_step = len(self.progress_blocks) - steps   # to go reverse in image sizes
        out = self.activation(self.rgb_layers[curr_step](x))

        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        downscaled = self.activation(self.rgb_layers[curr_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.progress_blocks[curr_step](out))
        out = self.fade_in(alpha, downscaled, out)

        for step in range(curr_step + 1, len(self.progress_blocks)):
            out = self.avg_pool(self.progress_blocks[step](out))

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)


if __name__ == "__main__":
    z_dim = 128
    in_channels = 256
    gen = Generator(z_dim, in_channels, img_channels=3)
    disc = Discriminator(in_channels, img_channels=3)

    for img_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        num_steps = int(math.log2(img_size / 4))
        x = torch.randn((1, z_dim, 1, 1))
        z = gen(x, alpha=0.5, steps=num_steps)
        assert z.shape == (1, 3, img_size, img_size)
        out = disc(z, alpha=0.2, steps=num_steps)
        assert out.shape == (1, 1)
        print("Success! At Image size: ", img_size)