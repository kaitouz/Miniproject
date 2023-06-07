import torch
from torch import nn, Tensor

##
def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)


class Encoder(nn.Module):
    def __init__(self, 
        input_size: tuple[int, int],
        num_input_channels : int,
        latent_vec_size: int,
        n_features: int,
        add_final_conv_layer: bool = True
    ) -> None:
        super().__init__()

        self.input_layers = nn.Sequential()
        self.input_layers.add_module(
            # (3, 64, (6, 4), 2, (4, 4))               3x192x128 --> 64x98x64
            f"initial-conv-{num_input_channels}-{64}",
            nn.Conv2d(num_input_channels, 64, kernel_size=(6, 4), stride=2, padding=(4, 4), bias=False),
        )
        self.input_layers.add_module(f"initial-relu-{n_features}", nn.LeakyReLU(0.2, inplace=True))

        self.pyramid_features = nn.Sequential()
        self.pyramid_features.add_module(
            # (64, 128, (6, 4), 2, (1, 1))               64x98x64 --> 128x48x32
            f"pyramid-{64}-{128}-conv",
            nn.Conv2d(64, 128, kernel_size=(6, 4), stride=2, padding=1, bias=False),
        )
        self.pyramid_features.add_module(f"pyramid-{128}-batchnorm", nn.BatchNorm2d(128))
        self.pyramid_features.add_module(f"pyramid-{128}-relu", nn.LeakyReLU(0.2, inplace=True))

        self.pyramid_features.add_module(
            # (128, 256, (6, 4), 2, (1, 1))               128x48x32 --> 256x23x16
            f"pyramid-{128}-{256}-conv",
            nn.Conv2d(128, 256, kernel_size=(6, 4), stride=2, padding=1, bias=False),
        )
        self.pyramid_features.add_module(f"pyramid-{256}-batchnorm", nn.BatchNorm2d(256))
        self.pyramid_features.add_module(f"pyramid-{256}-relu", nn.LeakyReLU(0.2, inplace=True))

        self.pyramid_features.add_module(
            # (256, 512, (6, 4), 2, (1, 1))               256x23x16 --> 512x10x8
            f"pyramid-{256}-{512}-conv",
            nn.Conv2d(256, 512, kernel_size=(6, 4), stride=2, padding=1, bias=False),
        )
        self.pyramid_features.add_module(f"pyramid-{512}-batchnorm", nn.BatchNorm2d(512))
        self.pyramid_features.add_module(f"pyramid-{512}-relu", nn.LeakyReLU(0.2, inplace=True))

        self.pyramid_features.add_module(
            # (512, 1024, (6, 4), 2, (1, 1))               512x10x8 --> 1024x4x4
            f"pyramid-{512}-{1024}-conv",
            nn.Conv2d(512, 1024, kernel_size=(6, 4), stride=2, padding=1, bias=False),
        )
        self.pyramid_features.add_module(f"pyramid-{1024}-batchnorm", nn.BatchNorm2d(1024))
        self.pyramid_features.add_module(f"pyramid-{1024}-relu", nn.LeakyReLU(0.2, inplace=True))

        if add_final_conv_layer:
            self.final_conv_layer = nn.Conv2d(
                1024,
                latent_vec_size,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            )

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Return latent vectors."""

        output = self.input_layers(input_tensor)
        output = self.pyramid_features(output)
        if self.final_conv_layer is not None:
            output = self.final_conv_layer(output)

        return output
        

class Decoder(nn.Module):
    def __init__(self, 
        input_size: tuple[int, int],
        num_input_channels : int,
        latent_vec_size: int,
        n_features: int,
    ) -> None:
        super().__init__()

        self.latent_input = nn.Sequential()
        self.latent_input.add_module(
            # (100, 1024, (4, 4), 1, (0, 0))               100x1x1 --> 1024x4x4
            f"initial-{latent_vec_size}-{1024}-convt",
            nn.ConvTranspose2d(latent_vec_size, 1024, kernel_size=(4, 4), stride=1, padding=0,bias=False),
        )
        self.latent_input.add_module(f"initial-{1024}-batchnorm", nn.BatchNorm2d(1024))
        self.latent_input.add_module(f"initial-{1024}-relu", nn.ReLU(True))

        self.inverse_pyramid = nn.Sequential()
        self.inverse_pyramid.add_module(
            # (1024, 512, (6, 4), 2, (1, 1))               1024x4x4 --> 512x10x8
            f"pyramid-{1024}-{512}-convt",
            nn.ConvTranspose2d(1024, 512, kernel_size=(6, 4), stride=2, padding=1, bias=False),
        )
        self.inverse_pyramid.add_module(f"pyramid-{512}-batchnorm", nn.BatchNorm2d(512))
        self.inverse_pyramid.add_module(f"pyramid-{512}-relu", nn.ReLU(True))
  
        self.inverse_pyramid.add_module(
            # (512, 256, (6, 4), 2, (1, 1))               512x10x8 --> 256x23x16
            f"pyramid-{512}-{256}-convt",
            nn.ConvTranspose2d(512, 256, kernel_size=(6, 4), stride=2, padding=1, bias=False),
        )
        self.inverse_pyramid.add_module(f"pyramid-{256}-batchnorm", nn.BatchNorm2d(256))
        self.inverse_pyramid.add_module(f"pyramid-{256}-relu", nn.ReLU(True))

        self.inverse_pyramid.add_module(
            # (512, 256, (6, 4), 2, (1, 1))               256x23x16 --> 128x48x32
            f"pyramid-{256}-{128}-convt",
            nn.ConvTranspose2d(256, 128, kernel_size=(6, 4), stride=2, padding=1, bias=False),
        )
        self.inverse_pyramid.add_module(f"pyramid-{128}-batchnorm", nn.BatchNorm2d(128))
        self.inverse_pyramid.add_module(f"pyramid-{128}-relu", nn.ReLU(True))


        self.inverse_pyramid.add_module(
            # (512, 256, (6, 4), 2, (1, 1))               128x48x32 --> 64x98x64
            f"pyramid-{128}-{64}-convt",
            nn.ConvTranspose2d(128, 64, kernel_size=(6, 4), stride=2, padding=1, bias=False),
        )
        self.inverse_pyramid.add_module(f"pyramid-{64}-batchnorm", nn.BatchNorm2d(64))
        self.inverse_pyramid.add_module(f"pyramid-{64}-relu", nn.ReLU(True))


        self.final_layers = nn.Sequential()
        self.final_layers.add_module(
            f"final-{64}-{num_input_channels}-convt",
            nn.ConvTranspose2d(64, num_input_channels, kernel_size=(6, 4), stride=2, padding=(0, 1), bias=False),
        )
        self.final_layers.add_module(f"final-{num_input_channels}-tanh", nn.Tanh())

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Return generated image."""
        output = self.latent_input(input_tensor)
        output = self.inverse_pyramid(output)
        output = self.final_layers(output)
        return output    
    

class Discriminator(nn.Module):

    def __init__(
        self, 
        input_size: tuple[int, int], 
        num_input_channels: int, 
        n_features: int
    ) -> None:
        super().__init__()
        
        encoder = Encoder(input_size=input_size, num_input_channels=num_input_channels, n_features=n_features, latent_vec_size=1)
        layers = []
        for block in encoder.children():
            if isinstance(block, nn.Sequential):
                layers.extend(list(block.children()))
            else:
                layers.append(block)

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module("Sigmoid", nn.Sigmoid())

    def forward(self, input_tensor: Tensor) -> tuple[Tensor, Tensor]:
        """Return class of object and features."""
        features = self.features(input_tensor)
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)
        return classifier, features
    

class Generator(nn.Module):

    def __init__(
        self,
        input_size: tuple[int, int],
        num_input_channels: int,
        latent_vec_size: int,
        n_features: int,
        add_final_conv_layer: bool = True,
    ) -> None:
        super().__init__()
        self.encoder1 = Encoder(input_size=input_size, num_input_channels=num_input_channels, latent_vec_size=latent_vec_size, n_features=n_features, add_final_conv_layer=add_final_conv_layer)
        self.decoder = Decoder(input_size=input_size, num_input_channels=num_input_channels, latent_vec_size=latent_vec_size, n_features=n_features)
        self.encoder2 = Encoder(input_size=input_size, num_input_channels=num_input_channels, latent_vec_size=latent_vec_size, n_features=n_features, add_final_conv_layer=add_final_conv_layer)

    def forward(self, input_tensor: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return generated image and the latent vectors."""
        latent_i = self.encoder1(input_tensor)
        gen_image = self.decoder(latent_i)
        latent_o = self.encoder2(gen_image)
        return gen_image, latent_i, latent_o