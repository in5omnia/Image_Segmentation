
import torch.nn as nn
import numpy as np
import torch
from dataset import *
from utils import *


#val_data = dataset("val/color", "val/label", target_transform=target_remap())
#test_data = dataset("rtest/color", "rtest/label", target_transform=target_remap())

#val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=diff_size_collate)
#test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True,collate_fn=diff_size_collate)

class EncoderBlock(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        # Added BatchNorm for potentially better reconstruction training
        self.encoder_block = nn.Sequential(
            nn.Conv2d(din, dout, kernel_size=3, padding=1, bias= False), # Bias often False before BN
            nn.BatchNorm2d(dout),
            nn.ReLU(),
            # Add another Conv layer 
            nn.Conv2d(dout, dout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dout),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)   
        )
    
    def forward(self, x):
        return self.encoder_block(x)


class Encoder(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.encoderPart1 = EncoderBlock(din, 64)
        self.encoderPart2 = EncoderBlock(64, 32)
        self.encoderPart3 = EncoderBlock(32, 16)
        # Store intermediate results if planning U-Net style skips later (Optional)
        self.features1 = None
        self.features2 = None
    
    def forward(self, x):
        self.features1 = self.encoderPart1(x)
        self.features2 = self.encoderPart2(self.features1)
        encoded_output = self.encoderPart3(self.features2)
        return encoded_output


class DecoderBlock(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.decoder_block = nn.Sequential(
            nn.ConvTranspose2d(din, dout, kernel_size=2, stride=2), # Upsamples H, W by 2
            nn.Conv2d(dout, dout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dout),
            nn.ReLU(inplace=True),
            nn.Conv2d(dout, dout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dout),
            nn.ReLU(inplace=True)
            # Above code instead of below
            #nn.Conv2d(din, dout, kernel_size=3, padding=1),
            #nn.ReLU(),
            #nn.Upsample(scale_factor=2)
        )
        
    def forward(self, x):
        return self.decoder_block(x)


class Decoder(nn.Module):
    def __init__(self, dout):
        super().__init__()
        self.decoderBlock1 = DecoderBlock(16, 16)
        self.decoderBlock2 = DecoderBlock(16, 32)
        self.decoderBlock3 = DecoderBlock(32, 64)
        #self.decoderOut = nn.Sequential(
        #    nn.Conv2d(64, dout, kernel_size=3, padding=1),
        #    nn.Sigmoid()
        #)
        
    def forward(self, x):
        x = self.decoderBlock1(x)
        x = self.decoderBlock2(x)
        x = self.decoderBlock3(x)
        #x = self.decoderOut(x)
        return x


class ReconstructionAutoencoder(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.encoder = Encoder(din)
        self.decoder = Decoder(dout)
        self.decoderOut = nn.Sequential(
            nn.Conv2d(64, dout, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        reconstructed = self.decoderOut(decoded)
        return reconstructed
        

class SegmentationEncoder(nn.Module):
    def __init__(self, din, pretrained_encoder_path=None, freeze_encoder=True):
        self.encoder = Encoder(din)
        
        # Load pre-trained weights (if path provided)
        if pretrained_encoder_path:
            try:
                # Load the state dict of the *entire* Autoencoder first
                full_autoencoder_state_dict = torch.load(pretrained_encoder_path, map_location=lambda storage, loc: storage) # Load to CPU initially

                # Create a new state dict for the encoder only
                encoder_state_dict = {}
                for key, value in full_autoencoder_state_dict.items():
                    if key.startswith('encoder.'):
                        # Remove the 'encoder.' prefix to match the keys in self.encoder
                        new_key = key[len('encoder.'):]
                        encoder_state_dict[new_key] = value

                # Load the extracted state dict into the encoder
                self.encoder.load_state_dict(encoder_state_dict, strict=True)
                print(f"Successfully loaded pre-trained encoder weights from {pretrained_encoder_path}")

            except FileNotFoundError:
                print(f"Warning: Pre-trained encoder file not found at {pretrained_encoder_path}. Encoder weights are random.")
            except Exception as e:
                print(f"Warning: Error loading pre-trained encoder weights: {e}. Encoder weights might be random.")
                # You might want `strict=False` if the architectures slightly differ, but it's risky.

        # Freeze the Encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder parameters frozen.")

    def forward(self, x):
        return self.encoder(x)  # Call the base class forward method


class SegmentationAutoencoder(nn.Module):
    def __init__(self, din, num_classes=4, pretrained_encoder_path=None):
        super().__init__()
        self.num_classes = num_classes

        # Initialize the Encoder
        self.encoder = SegmentationEncoder(din, pretrained_encoder_path=pretrained_encoder_path, freeze_encoder=True)

        # Initialize the Decoder
        self.decoder = Decoder(dout=self.num_classes)
        # Decoder parameters have requires_grad = True by default

        # Final convolution to get scores for each class per pixel
        self.finalConv = nn.Conv2d(64, num_classes, kernel_size=1)
        # No Sigmoid/Softmax here because CrossEntropyLoss expects raw logits

    def forward(self, x):
        # Pass input through the frozen encoder
        encoded_features = self.encoder(x)

        # Pass the encoded features through the trainable segmentation decoder
        segmentation_output = self.decoder(encoded_features)

        # Final convolution to get logits for each class
        segmentation_output = self.finalConv(segmentation_output)

        return segmentation_output # Shape: (B, num_classes, H, W)
    




class SegmentationEncoder(nn.Module):
    def __init__(self, din, pretrained_encoder_path=None, freeze_encoder=True):
        super().__init__()
        self.encoder = Encoder(din)
        
        # Load pre-trained weights (if path provided)
        if pretrained_encoder_path:
            try:
                # Load the state dict of the *entire* Autoencoder first
                full_autoencoder_state_dict = torch.load(pretrained_encoder_path, map_location=lambda storage, loc: storage) # Load to CPU initially
                try:
                    model_state_dict = full_autoencoder_state_dict["model_state_dict"]
                except Exception as e:
                    print(f"Warning: Could not load model state dict from checkpoint: {e}. ")

                # Create a new state dict for the encoder only
                encoder_state_dict = {}
                for key, value in model_state_dict.items():
                    if key.startswith('encoder.'):
                        print(key)
                        # Remove the 'encoder.' prefix to match the keys in self.encoder
                        new_key = key[len('encoder.'):]
                        encoder_state_dict[new_key] = value

                # Load the extracted state dict into the encoder
                self.encoder.load_state_dict(encoder_state_dict, strict=True)
                print(f"Successfully loaded pre-trained encoder weights from {pretrained_encoder_path}")

            except FileNotFoundError:
                print(f"Warning: Pre-trained encoder file not found at {pretrained_encoder_path}. Encoder weights are random.")
            except Exception as e:
                print(f"Warning: Error loading pre-trained encoder weights: {e}. Encoder weights might be random.")
                # You might want `strict=False` if the architectures slightly differ, but it's risky.

        # Freeze the Encoder parameters
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        print("Encoder parameters frozen.")

    def forward(self, x):
        return self.encoder(x)  # Call the base class forward method


class SegmentationAutoencoder(nn.Module):
    def __init__(self, din, num_classes=4, pretrained_encoder_path=None, freeze_encoder=True):
        super().__init__()
        self.num_classes = num_classes

        # Initialize the Encoder
        self.encoder = SegmentationEncoder(din, pretrained_encoder_path=pretrained_encoder_path, freeze_encoder=True)

        # Initialize the Decoder
        self.decoder = Decoder(dout=self.num_classes)
        # Decoder parameters have requires_grad = True by default

        # Final convolution to get scores for each class per pixel
        self.finalConv = nn.Conv2d(64, num_classes, kernel_size=1)
        # No Sigmoid/Softmax here because CrossEntropyLoss expects raw logits

    def forward(self, x):
        # Pass input through the frozen encoder
        encoded_features = self.encoder(x)

        # Pass the encoded features through the trainable segmentation decoder
        segmentation_output = self.decoder(encoded_features)

        # Final convolution to get logits for each class
        segmentation_output = self.finalConv(segmentation_output)

        return segmentation_output # Shape: (B, num_classes, H, W)
    