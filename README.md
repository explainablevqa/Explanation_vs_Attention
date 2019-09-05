# Explanation vs Attention : A Two Player Game to obtain Attention for VQA

## Training Steps:

### A. Preprocessing:

  --- Image Preprocessing 
       --- python preprocess-images.py
  ---Create VoCabulary     
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
