from clip.clipunet import ClipUNet
from unet.unet import unet
import torch
from torch import nn

class PromptModel(nn.Module):
    """
    Initializes the PromptModel.
    Args:
        path (str, optional): Path to the checkpoint file. Defaults to None.
    """
    def __init__(self, path=None):
        super().__init__()
        
        self.clip = ClipUNet()
        self.mask = unet(4, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        
        if path is not None:
            try:
                # Load the checkpoint.
                checkpoint = torch.load(path, weights_only=False, map_location=lambda storage, loc: storage) # Load to CPU initially
                self.clip.load_state_dict(checkpoint["model_state_dict"])
            except Exception as e:
                print(f"Error loading checkpoint: {str(e)[:200]}")
                raise
        
        # Freeze the clip parameters.
        for param in self.clip.parameters():
            param.requires_grad = False
    
    def forward(self, x, heatmap):
        # Pass input through clip model.
        clip_logit = self.clip(x)
        # Apply softmax to clip logits.
        clip_prob = self.softmax(clip_logit)
        
        # Concatenate input and heatmap, pass through mask model.
        mask_logit = self.mask(torch.concat([x, heatmap], dim=1))
        # Apply sigmoid to mask logits.
        mask_prob = self.sigmoid(mask_logit)
        
        # Initialize tensor to store final probabilities.
        final_probs = torch.empty_like(clip_prob)
        # Calculate selected probabilities.
        selected_prob = mask_prob * clip_prob

        # Assign probabilities to the final tensor.
        # Assign the selected probabilities for background, cat, and dog.
        final_probs[:, 1:4, :, :] = selected_prob[:, 0:3, :, :]
        # Assign the mask probability for deactivated class.
        final_probs[:, 0:1, :, :] = 1.0 - mask_prob
        # Merge boundary class with background
        final_probs[:, 1:2, :, :] += selected_prob[:, 3:4, :, :]

        return final_probs