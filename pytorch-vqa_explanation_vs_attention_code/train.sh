
#To Train Adversial enable this 
CUDA_VISIBLE_DEVICES=5 python -u train.py |tee log_train_adv.txt

#To Train MMD enable this 
# CUDA_VISIBLE_DEVICES=5 python -u train_mmd.py |tee log_train_mms.txt

#To Train Coral enable this 
# CUDA_VISIBLE_DEVICES=5 python -u train_coral.py |tee log_train_coral.txt

#To Train MSE enable this 
# CUDA_VISIBLE_DEVICES=5 python -u train_mse.py |tee log_train_mse.txt




