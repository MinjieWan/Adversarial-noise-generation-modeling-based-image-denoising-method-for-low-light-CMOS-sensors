# Adversarial-noise-generation-modeling-based-image-denoising-method-for-low-light-CMOS-sensors

Official PyTorch implementations of FDA adversarial noise generation model and FD-UNet.

---

## ⚙️ Requirements

- Python==3.10  
- torch==2.5.1  
- torchvision==0.20.1  
- opencv-python==4.10.0  
- pillow==11.3.0  
- scikit-image==0.20.0  
- tensorboard==2.20.0
- tqdm==4.67.1
- einops==0.8.1
---

## 📂 Dataset

The full dataset  are not included in this repository due to their large file size.  

Please download them manually and place them in the following structure.

- Dataset： https://pan.baidu.com/s/18uynyJvgBpRNAHEPfXfI1A?pwd=msfr 

Expected structure:

```
noise_generation/
└── train/
    ├── input2/
    ├── gt2/
    
FD-Unet/
└── train/
    ├── ours_0227/
└── test/
    ├── input2/
    ├── gt2/
```


---

## 📁 Project Structure

```bash
Adversarial-noise-generation-modeling-based-image-denoising-method-for-low-light-CMOS-sensors/
├── FD-Unet/
├── FDUNet_model/
│   └── ours_0227
         ├── FDUNET.pth.tar/
│   ├── datasets/             # Data preprocessing and loading
│   ├── models/               # network architecture
│   ├── train.py              # training script
│   ├── infer.py              # testing script
│   └── utils.py              # model processing tools
│
└── noise_generation/
    ├── datasets/             # Data preprocessing and loading
    ├── FDUNet/
        └── FDUNet_model/
            └── ref_0221
                ├── best_model.pth/
    ├── generation_model/
        └── 2026-02-23
            └── 13-52-09
                ├── G.pth/
    ├── models/               # network architecture
    ├── config.py             # Parameter settings
    ├── infer.py              # testing script
    └── train.py              # training script
```

## 🚀 Training & Testing

Enable scripts in the noise_generation folder.
### Training
python train.py

You can modify training configurations in config.py, including:

- dataset path
- batch size
- learning rate
- number of epochs
- Loss weight

### Testing
python infer.py

Enable scripts in the FD-Unet folder.

### Training
python train.py

You can modify training configurations in train.py, including:

- dataset path
- batch size
- learning rate
- number of epochs

### Testing
python infer.py
## ⚠️ Notes:

- Please download the model path from the link before testing.


## 📦 Pre-trained Models

For noise_generation, We provide trained models in the generation_model/ and FDUNet/ directory:
├── noise_generation/
    └── FDUNet/
        └── FDUNet_model/
            └── ref_0221
                ├── best_model.pth/
└── noise_generation/
    └── generation_model/
        └── 2026-02-23
              └── 13-52-09
                  ├── G.pth/
                  
For FD-Unet, please download the trained model and place it in the following directory:

```
├── FD-Unet/
    └── FDUNet_model/
        └── ours_0227
             ├── FDUNET.pth.tar/
```
- Pre-trained model: https://pan.baidu.com/s/1L-37kO5Vd4Q9UihnZAZXUg?pwd=ccue 

