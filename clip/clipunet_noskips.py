import torch.nn as nn
from transformers import CLIPVisionModel, CLIPVisionConfig

PRETRAINED_MODEL_NAME = "openai/clip-vit-base-patch16"

class ClipViTEncoderNoSkips(nn.Module):
    """
    This class implements a CLIP ViT encoder without skip connections.

    Args:
        model_name (str, optional): The name of the pre-trained CLIP model to use. Defaults to "openai/clip-vit-base-patch16".
        freeze_encoder (bool, optional): Whether to freeze the encoder weights. Defaults to True.

    Returns:
        torch.Tensor: The bottleneck features of the input image.
    """
    def __init__(self, model_name="openai/clip-vit-base-patch16", freeze_encoder=True): # Removed bottleneck_index
        super().__init__()

        self.config = CLIPVisionConfig.from_pretrained(model_name)
        self.clip_vit = CLIPVisionModel.from_pretrained(model_name)

        if freeze_encoder:
            # Freeze the encoder parameters if specified
            for param in self.clip_vit.parameters():
                param.requires_grad = False

        self.grid_size = self.config.image_size // self.config.patch_size
        self.hidden_dim = self.config.hidden_size

    def forward(self, x):
        if x.shape[2] != self.config.image_size or x.shape[3] != self.config.image_size:
             # Warn the user if the input image size doesn't match the expected size
             print(
                 f"Input image size ({x.shape[2]}x{x.shape[3]}) doesn't match "
                 f"CLIP expected size ({self.config.image_size}x{self.config.image_size}). "
                 f"Behavior may be unexpected. Consider resizing input."
             )

        outputs = self.clip_vit(pixel_values=x)
        last_hidden_state = outputs.last_hidden_state
        
        patch_embeddings = last_hidden_state[:, 1:, :] # Remove CLS token
        bottleneck_features = patch_embeddings \
                                .reshape(x.shape[0], self.grid_size, self.grid_size, self.hidden_dim) \
                                .permute(0, 3, 1, 2).contiguous()
        
        return bottleneck_features


class DecoderBlockNoSkip(nn.Module):
    """
    This class implements a decoder block without skip connections.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Returns:
        torch.Tensor: Output tensor after the decoder block.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Upsample the input feature map
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2) # Maybe reduce channels here
        
        # Convolutional block
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), # Adjusted input channels here
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv_block(x)
        return x


class UNetDecoderNoSkips(nn.Module):
    """
    This class implements a UNet decoder without skip connections.

    Args:
        encoder_hidden_dim (int): The hidden dimension of the encoder.
        decoder_channels (list of int): The number of channels for each decoder block.

    Returns:
        torch.Tensor: The output tensor after passing through the decoder.
    """
    def __init__(self, encoder_hidden_dim, decoder_channels):
        super().__init__()

        # Initial convolution layer
        self.init_conv = nn.Conv2d(encoder_hidden_dim, decoder_channels[0], kernel_size=1)

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        in_channels = decoder_channels[0]
        for i in range(len(decoder_channels)-1):
            out_ch = decoder_channels[i+1]
            
            block = DecoderBlockNoSkip(
                in_channels=in_channels,
                out_channels=out_ch,
            )
            self.decoder_blocks.append(block)
            in_channels = out_ch

    def forward(self, x):
        x = self.init_conv(x)
        for block in self.decoder_blocks:
            x = block(x)
        return x


class ClipUNetNoSkips(nn.Module):
    """
    This class implements a CLIP UNet without skip connections.

    Args:
        num_classes (int, optional): The number of output classes. Defaults to 4.
        decoder_channels (list of int, optional): The number of channels for each decoder block. Defaults to [1024, 512, 256, 128, 64].
        freeze_encoder (bool, optional): Whether to freeze the encoder weights. Defaults to True.
        model_name (str, optional): The name of the pre-trained CLIP model to use. Defaults to "openai/clip-vit-base-patch16".

    Returns:
        torch.Tensor: The output tensor after passing through the UNet.
    """
    def __init__(self,
                 num_classes=4,
                 decoder_channels=[1024, 512, 256, 128, 64],
                 freeze_encoder=True,
                 model_name="openai/clip-vit-base-patch16"
                 ):
        super().__init__()

        # Encoder
        self.encoder = ClipViTEncoderNoSkips(
            model_name=model_name,
            freeze_encoder=freeze_encoder
        )
        
        # Decoder
        self.decoder = UNetDecoderNoSkips(
            encoder_hidden_dim=self.encoder.hidden_dim,
            decoder_channels=decoder_channels
        )
        # Output layer
        self.output_layer = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        decoder_output = self.decoder(x)
        output = self.output_layer(decoder_output)
        return output