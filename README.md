# Explanation vs Attention : A Two Player Game to obtain Attention for VQA

## Training Steps:
We have provide training process for basic attention model using  VGG16 and  Resnet150 image feature.
For Bottom-up feature, we need to download from bottom up repo and need to change little bit for bounding box in the training file. 

### A. Preprocessing:

  --- Image Preprocessing: Need to change pretrained model for VGG16, Resnet150, and Bottom-up Feature.
            
       --- python preprocess-images.py
  
  --- Create VoCabulary:

    --- python preprocess-vocab.py

### B. To train the model with:
      --- ./train.sh
      
### C. To Evaluate the model with:
      --- ./evaluate.sh

### D. To plot the training progress with log file:
      --- python view-log.py <path to .pth log>

### E. Python 3 dependencies
      torch
      torchvision
      h5py
      tqdm
