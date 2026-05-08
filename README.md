# Adversarial-noise-generation-modeling-based-image-denoising-method-for-low-light-CMOS-sensors

Official PyTorch implementations of FDA adversarial noise generation model and FD-UNet.

---

## ⚙️ Requirements

- Python==3.10  
- torch==2.5  
- torchvision==0.20.1  
- numpy==1.26.4  
- numpy  
- scikit-image  
- pytorch-msssim  

---

## 📂 Dataset

We use the following datasets:

### NUDT-MIRSDT
- Download: https://pan.baidu.com/s/1pSN350eurMafLiHBQBnrPA  
- Code: `5whn`

### TSIRMT
- Download: https://drive.google.com/drive/folders/1aWDNdUWkTOuV3fILbgLDEqM2N2erW05n  

⚠️ **Note:**
- The `dataset/` folder in this repository only contains **partial samples**.  
- Please download the **full datasets manually** from the links above.

---

## 📁 Project Structure

```bash
JASTFR-Net/
├── dataset/        # sample data (not full dataset)
├── checkpoints/    # pre-trained models
├── JASTFRNet.py    # network definition
├── train.py        # training script
└── test_xxx.py     # testing script
```

## 🚀 Training & Testing

### Training
python train.py

You can modify training configurations in train.py, including:

- dataset path
- batch size
- learning rate
- number of epochs

### Testing
python test_xxx.py

## ⚠️ Notes:

- Please modify the model path in test_xxx.py before testing.
- Different datasets correspond to different test scripts.
- Select the appropriate test_xxx.py according to the dataset you use.

## 📦 Pre-trained Models

We provide trained models in the checkpoints/ directory:

- NUDT-MIRSDT → NUDT.pth
- TSIRMT → TSIRMT.pth

Please load the corresponding model when testing on different datasets.
