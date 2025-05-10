import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPVisionConfig


class ClipViTEncoder(nn.Module):
    """
    Encodes an image using CLIP's Vision Transformer.

    Args:
        model_name (str): The name of the pre-trained CLIP model to use.
        freeze_encoder (bool): Whether to freeze the CLIP encoder's parameters.
        skip_indices (list): The indices of the hidden states to use for skip connections.

    Returns:
        bottleneck_features: The bottleneck features from the encoder.
        skip_features_list: A list of skip features from the encoder.
    """
    def __init__(self, model_name="openai/clip-vit-base-patch16", freeze_encoder=True, skip_indices=[3, 5, 7, 9]):
        super().__init__()

        self.skip_indices = sorted(skip_indices)

        self.config = CLIPVisionConfig.from_pretrained(model_name)
        self.clip_vit = CLIPVisionModel.from_pretrained(model_name)

        if freeze_encoder:
            for param in self.clip_vit.parameters():
                param.requires_grad = False

        self.grid_size = self.config.image_size // self.config.patch_size
        self.hidden_dim = self.config.hidden_size

    def forward(self, x):
        if x.shape[2] != self.config.image_size or x.shape[3] != self.config.image_size:
             print(
                 f"Input image size ({x.shape[2]}x{x.shape[3]}) doesn't match "
                 f"CLIP expected size ({self.config.image_size}x{self.config.image_size}). "
                 f"Behavior may be unexpected. Consider resizing input."
             )

        outputs = self.clip_vit(pixel_values=x, output_hidden_states=True)
        all_hidden_states = outputs.hidden_states
        last_hidden_state = outputs.last_hidden_state

        # (N, 49, 768) -> (N, 7, 7, 768) -> (N, 768, 7, 7) = (N, C, H, W)
        patch_embeddings = last_hidden_state[:, 1:, :] # Remove CLS
        bottleneck_features = patch_embeddings \
                                .reshape(x.shape[0], self.grid_size, self.grid_size, self.hidden_dim) \
                                .permute(0, 3, 1, 2).contiguous()

        skip_features_list = []
        for i in self.skip_indices:
            hidden_state = all_hidden_states[i]
            patch_embeddings = hidden_state[:, 1:, :] # Remove CLS

            # (N, 49, 768) -> (N, 7, 7, 768) -> (N, 768, 7, 7) = (N, C, H, W)
            reshaped_features = patch_embeddings \
                                    .reshape(x.shape[0], self.grid_size, self.grid_size, self.hidden_dim) \
                                    .permute(0, 3, 1, 2).contiguous()

            skip_features_list.append(reshaped_features)

        return bottleneck_features, skip_features_list


class DecoderBlock(nn.Module):
    """
    A single decoder block for the UNet.

    Args:
        in_channels (int): The number of input channels.
        in_channels_skip (int): The number of channels from the skip connection.
        out_channels (int): The number of output channels.

    Returns:
        x (torch.Tensor): The output tensor after the decoder block.
    """
    def __init__(self, in_channels, in_channels_skip, out_channels):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.skip_conv = nn.Conv2d(in_channels_skip, in_channels // 2, kernel_size=1) # 768 channels to in_channels//2 (eg. 512)

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        skip = self.skip_conv(skip)

        if skip.shape[2:] != x.shape[2:]:
              skip= F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)

        x = self.conv_block(x)
        return x


class UNetDecoder(nn.Module):
    """
    The UNet decoder.

    Args:
        encoder_hidden_dim (int): The hidden dimension of the encoder.
        decoder_channels (list): A list of the number of channels for each decoder block.

    Returns:
        x (torch.Tensor): The output tensor after the decoder.
    """
    def __init__(self, encoder_hidden_dim, decoder_channels):
        super().__init__()

        self.init_conv = nn.Conv2d(encoder_hidden_dim, decoder_channels[0], kernel_size=1)

        self.decoder_blocks = nn.ModuleList()
        in_channels = decoder_channels[0]
        for i in range(len(decoder_channels)-1):
            out_ch = decoder_channels[i+1]

            block = DecoderBlock(
                in_channels=in_channels,
                in_channels_skip=encoder_hidden_dim,
                out_channels=out_ch
            )
            self.decoder_blocks.append(block)

            in_channels = out_ch


    def forward(self, x, skips):
        x = self.init_conv(x) # 768 -> 1024 channels
        for block, skip in zip(self.decoder_blocks, reversed(skips)):
            x = block(x, skip)

        return x


class ClipUNet(nn.Module):
    """
    The main ClipUNet model.

    Args:
        num_classes (int): The number of output classes.
        decoder_channels (list): A list of the number of channels for each decoder block.
        freeze_encoder (bool): Whether to freeze the CLIP encoder's parameters.
        model_name (str): The name of the pre-trained CLIP model to use.
        skip_indices (list): The indices of the hidden states to use for skip connections.

    Returns:
        output (torch.Tensor): The output tensor.
    """
    def __init__(self,
                 num_classes=4,
                 decoder_channels=[1024, 512, 256, 128, 64],
                 freeze_encoder=True,
                 model_name="openai/clip-vit-base-patch16",
                 skip_indices=[3, 5, 7, 9]
                 ):
        super().__init__()

        self.encoder = ClipViTEncoder(
            model_name=model_name,
            freeze_encoder=freeze_encoder,
            skip_indices=skip_indices
        )

        self.decoder = UNetDecoder(
            encoder_hidden_dim=self.encoder.hidden_dim,
            decoder_channels=decoder_channels
        )

        self.output_layer = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)


    def forward(self, x):
        x, skips = self.encoder(x)
        decoder_output = self.decoder(x, skips)
        output = self.output_layer(decoder_output)
        return output