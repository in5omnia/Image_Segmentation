# Semantic Image Segmentation with Deep Learning
This repository contains code and experiments for a **Computer Vision coursework project** at the University of Edinburgh.  
The project investigates deep learning methods for **semantic image segmentation** on the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/%7Evgg/data/pets/).  
We compared multiple architectures, including classic CNN-based models, transfer learning with CLIP, and an interactive prompt-based approach. 
This project was done with [@chengandre](https://github.com/chengandre).

---

## Project Overview

We implemented and evaluated the following models:

- **U-Net**  
  Encoder–decoder CNN with skip connections, trained end-to-end.  

- **Autoencoder + Segmentation**  
  A self-supervised autoencoder pre-trained on reconstruction, reusing its encoder for segmentation.  

- **CLIP-based Segmentation**  
  Uses frozen **CLIP ViT** embeddings as features with a custom U-Net–style decoder.  

- **Prompt-based Interactive Segmentation**  
  Extends the CLIP model to allow user-guided point prompts for interactive segmentation.  


---

## Results Summary

| Model                  | Accuracy | Dice  | IoU   |
|-------------------------|----------|-------|-------|
| UNet (Augmented)        | 0.9462   | 0.8661| 0.7687|
| UNet (No Aug)           | 0.9444   | 0.8632| 0.7643|
| CLIP (Augmented)        | 0.9732   | 0.9442| 0.8946|
| CLIP (No Aug)           | 0.9723   | 0.9414| 0.8897|
| Autoencoder             | 0.8712   | 0.6804| 0.5382|
| Prompt-based (Fine-tuned)| 0.8321  | 0.7088| 0.5497|

Key findings:
- **CLIP-based models** outperform others by leveraging large-scale pretraining.  
- **U-Net** is competitive but less robust.  
- **Autoencoder features** did not transfer well to segmentation.  
- **Prompt-based models** enable interaction but are harder to benchmark directly.  
- **Data augmentation** improved robustness more than clean test set accuracy.  

---

## Notes

- This repository was developed as **coursework** and is not yet packaged for reuse.  
- Code may require adjustments to run outside the original Kaggle/Colab environment.  
- Training and evaluation scripts are included but not fully documented.
- Both contributors worked on model design, implementation, experiments, and report writing.

