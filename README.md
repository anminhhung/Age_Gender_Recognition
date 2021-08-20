## 1. Setup params
Opens file **config.ini**:
```python
[device]
device = cuda 

[dataset]
data_path = mega_age_gender
train_val_ratio = 0.2 
max_age = 69
    
[callback]
mode = min 
monitor = val_loss
 # checkpoint_path
checkpoint_path = model_checkpoint
# save top k best models
save_top_k = 3 
# path file: checkpoint_path/filename.ckpt
filename = AgeGender-{epoch:02d}-{val_loss:.2f} 

[model]
epochs = 5
lr = 0.001 
batch_size = 32

[inference]
checkpoint_path = models/best_checkpoint.ckpt
gender_threshold = 0.5
```

Setup and save file.

---
## 2. Training
    python3 train.py --cfg_path configs/config.ini

---
## 3. Predict
    python3 predict.py --cfg_path configs/config.ini --image dataset/0_A0_G0.png