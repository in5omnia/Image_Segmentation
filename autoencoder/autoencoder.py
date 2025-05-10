import torch.nn as nn
import numpy as np
import torch


class EncoderBlock(nn.Module):
    """
    A block for the encoder, consisting of two convolutional layers, batch normalization, ReLU activations, and max pooling.
    Args:
        din (int): Number of input channels.
        dout (int): Number of output channels.
    Returns:
        tuple: A tuple containing the pooled output and the skip connection features.
    """
    def __init__(self, din, dout):
        super().__init__()
        self.conv1 = nn.Conv2d(din, dout, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dout)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(dout, dout, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dout)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        skip_connection = self.relu2(x) # Features before pooling
        pooled_output = self.pool(skip_connection)
        return pooled_output, skip_connection

class Encoder(nn.Module):
    """
    The encoder module composed of multiple encoder blocks.
    Args:
        din (int): Number of input channels.
        base_channels (int): Base number of channels to be used for the encoder blocks.
    Returns:
        tuple: A tuple containing the bottleneck features and skip connections from each encoder block.
    """
    def __init__(self, din, base_channels):
        super().__init__()
        self.encoderPart1 = EncoderBlock(din, base_channels)
        self.encoderPart2 = EncoderBlock(base_channels, base_channels*2)
        self.encoderPart3 = EncoderBlock(base_channels*2, base_channels*4)

    def forward(self, x):
        x1_pooled, skip1 = self.encoderPart1(x)
        x2_pooled, skip2 = self.encoderPart2(x1_pooled)
        bottleneck, skip3 = self.encoderPart3(x2_pooled)
        return bottleneck, skip3, skip2, skip1


class DecoderBlockWithSkips(nn.Module):
    """
    A decoder block that uses skip connections.
    Args:
        din_up (int): Number of input channels from the previous upsampled layer.
        din_skip (int): Number of input channels from the skip connection.
        dout (int): Number of output channels.
    Returns:
        torch.Tensor: Output tensor after upsampling, concatenation, and convolution.
    """
    def __init__(self, din_up, din_skip, dout):
        super().__init__()
        self.up = nn.ConvTranspose2d(din_up, dout, kernel_size=2, stride=2)
        conv_input_channels = dout + din_skip # Input to convs includes skip features
        self.convs = nn.Sequential(
            nn.Conv2d(conv_input_channels, dout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dout),
            nn.ReLU(inplace=True),
            nn.Conv2d(dout, dout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dout),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_features):
        x_upsampled = self.up(x)
        if skip_features.shape[2:] != x_upsampled.shape[2:]:
            diffY = skip_features.size()[2] - x_upsampled.size()[2]
            diffX = skip_features.size()[3] - x_upsampled.size()[3]
            if diffY < 0 or diffX < 0: 
                raise ValueError("Upsampled larger than skip")
            skip_features = skip_features[:, :, diffY // 2 : diffY // 2 + x_upsampled.size()[2],
                                           diffX // 2 : diffX // 2 + x_upsampled.size()[3]]


        x_concat = torch.cat([x_upsampled, skip_features], dim=1)
        output = self.convs(x_concat)
        return output


class DecoderWithSkips(nn.Module):
    """
    The decoder module with skip connections.
    Args:
        base_channels (int): Base number of channels.
    Returns:
        torch.Tensor: Output tensor after decoding.
    """
    def __init__(self, base_channels):
        super().__init__()
        self.decoderBlock1 = DecoderBlockWithSkips(din_up=base_channels*4, din_skip=base_channels*4, dout=base_channels*2)
        self.decoderBlock2 = DecoderBlockWithSkips(din_up=base_channels*2, din_skip=base_channels*2, dout=base_channels)
        self.decoderBlock3 = DecoderBlockWithSkips(din_up=base_channels, din_skip=base_channels, dout=base_channels)

    def forward(self, bottleneck, skip3, skip2, skip1):
        d1 = self.decoderBlock1(bottleneck, skip3)
        d2 = self.decoderBlock2(d1, skip2)
        d3 = self.decoderBlock3(d2, skip1)
        return d3 # Output feature map (B, base_channels, H, W)


class DecoderBlockNoSkips(nn.Module):
    """
    A decoder block without skip connections.
    Args:
        din_up (int): Number of input channels from the previous upsampled layer.
        dout (int): Number of output channels.
    Returns:
        torch.Tensor: Output tensor after upsampling and convolution.
    """
    def __init__(self, din_up, dout):
        super().__init__()
        # Upsample and change channels
        self.up = nn.ConvTranspose2d(din_up, dout, kernel_size=2, stride=2)
        # Convolutions only process the upsampled features
        # Input channels to convs is just 'dout' (output channels of self.up)
        self.convs = nn.Sequential(
            nn.Conv2d(dout, dout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dout),
            nn.ReLU(inplace=True),
            nn.Conv2d(dout, dout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dout),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Only takes features from the previous layer 'x'
        x_upsampled = self.up(x)
        # No concatenation
        output = self.convs(x_upsampled)
        return output


class DecoderNoSkips(nn.Module):
    """
    The decoder module without skip connections.
    Args:
        base_channels (int): Base number of channels.
    Returns:
        torch.Tensor: Output tensor after decoding.
    """
    def __init__(self, base_channels):
        super().__init__()
        self.decoderBlock1 = DecoderBlockNoSkips(din_up=base_channels*4, dout=base_channels*2) # 256 -> 128
        self.decoderBlock2 = DecoderBlockNoSkips(din_up=base_channels*2, dout=base_channels)   # 128 -> 64
        self.decoderBlock3 = DecoderBlockNoSkips(din_up=base_channels, dout=base_channels)       # 64 -> 64

    def forward(self, bottleneck):
        # Only takes the bottleneck as input
        d1 = self.decoderBlock1(bottleneck)
        d2 = self.decoderBlock2(d1)
        d3 = self.decoderBlock3(d2)
        return d3 # Output feature map (B, base_channels, H, W)


class ReconstructionAutoencoder(nn.Module):
    """
    A reconstruction autoencoder model.
    Args:
        din (int): Number of input channels.
        dout (int): Number of output channels.
        base_channels (int): Base number of channels.
    Returns:
        torch.Tensor: Reconstructed output tensor.
    """
    def __init__(self, din, dout=3, base_channels=64):
        super().__init__()
        self.encoder = Encoder(din, base_channels)
        # No skips in Reconstruction AE
        self.decoder = DecoderNoSkips(base_channels)

        # Final layer maps DecoderSimple output (base_channels=64) to reconstruction
        self.decoderOut = nn.Sequential(
            nn.Conv2d(base_channels, dout, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encode, getting bottleneck and skip connections
        bottleneck, skip3_ignored, skip2_ignored, skip1_ignored = self.encoder(x)
        # Decode using ONLY the bottleneck with the simple decoder
        decoded_features = self.decoder(bottleneck)
        # Apply final layer
        reconstructed = self.decoderOut(decoded_features)
        return reconstructed


class SegmentationEncoder(nn.Module):
    """
    An encoder module for segmentation tasks, with optional pre-trained weights and freezing.
    Args:
        din (int): Number of input channels.
        base_channels (int): Base number of channels.
        pretrained_encoder_path (str, optional): Path to the pre-trained encoder weights. Defaults to None.
        freeze_encoder (bool, optional): Whether to freeze the encoder parameters. Defaults to True.
    Returns:
        tuple: Bottleneck and skip connections from the encoder.
    """
    def __init__(self, din, base_channels, pretrained_encoder_path=None, freeze_encoder=True):
        super().__init__()
        self.encoder = Encoder(din, base_channels)

        if pretrained_encoder_path:
            try:
                full_state_dict = torch.load(pretrained_encoder_path, weights_only=False, map_location=lambda storage, loc: storage)
                # Handle potential checkpoint structure variations
                if "model_state_dict" in full_state_dict: 
                    model_state_dict = full_state_dict["model_state_dict"]
                elif "state_dict" in full_state_dict: 
                    model_state_dict = full_state_dict["state_dict"]
                else: 
                    model_state_dict = full_state_dict # Assume it's the state dict directly

                encoder_state_dict = {}
                has_encoder_prefix = any(k.startswith('encoder.') for k in model_state_dict.keys())

                for key, value in model_state_dict.items():
                    if has_encoder_prefix:
                         if key.startswith('encoder.'):
                             new_key = key[len('encoder.'):]
                             encoder_state_dict[new_key] = value

                if not encoder_state_dict:
                     print("Warning: Could not extract encoder state dict. Checkpoint might be empty or incompatible.")
                else:
                    load_result = self.encoder.load_state_dict(encoder_state_dict, strict=True) # Use strict=False for robustness
                    print(f"Loaded encoder weights. Load result:")
                    if load_result.missing_keys: 
                        print("  Missing keys:", load_result.missing_keys)
                    if load_result.unexpected_keys: 
                        print("  Unexpected keys:", load_result.unexpected_keys)
                    if not load_result.missing_keys and not load_result.unexpected_keys: 
                        print("  All keys matched successfully.")

            except FileNotFoundError:
                print(f"Warning: Pre-trained encoder file not found: {pretrained_encoder_path}. Using random weights.")
            except Exception as e:
                print(f"Warning: Error loading weights: {e}. Check compatibility. Using random weights.")

        if freeze_encoder:
            if not pretrained_encoder_path:
                print("Warning: Freezing encoder, but no pre-trained weights were loaded.")
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Encoder parameters frozen.")
        else:
            print("Encoder parameters are trainable.")


    def forward(self, x):
        # Returns bottleneck, skip3, skip2, skip1
        return self.encoder(x)



class SegmentationAutoencoder(nn.Module):
    """
    A segmentation autoencoder model.
    Args:
        din (int): Number of input channels.
        base_channels (int): Base number of channels.
        num_classes (int): Number of output classes.
        pretrained_encoder_path (str, optional): Path to the pre-trained encoder weights. Defaults to None.
        freeze_encoder (bool, optional): Whether to freeze the encoder parameters. Defaults to True.
    Returns:
        torch.Tensor: Segmentation logits.
    """
    def __init__(self, din, base_channels=64, num_classes=4, pretrained_encoder_path=None, freeze_encoder=True):
        super().__init__()
        self.num_classes = num_classes

        # Initialize the Encoder (via wrapper for loading/freezing)
        self.encoder = SegmentationEncoder(din, base_channels, pretrained_encoder_path=pretrained_encoder_path, freeze_encoder=freeze_encoder)

        # Use the Decoder WITH Skips for Segmentation
        self.decoder = DecoderWithSkips(base_channels)

        # Final convolution maps DecoderWithSkips output (base_channels=64) to class scores
        self.finalConv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # 1. Encoder gets bottleneck and skips
        bottleneck, skip3, skip2, skip1 = self.encoder(x)

        # 2. Decoder uses bottleneck AND skips
        decoder_output = self.decoder(bottleneck, skip3, skip2, skip1)

        # 3. Final 1x1 convolution for class logits
        segmentation_logits = self.finalConv(decoder_output)

        return segmentation_logits