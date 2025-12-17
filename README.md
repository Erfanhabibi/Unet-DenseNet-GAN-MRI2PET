# 🧠 U-Net/DenseNet GAN for MRI-to-PET Synthesis

A PyTorch implementation of a conditional GAN (cGAN) that synthesizes PET images from MRI scans using a U-Net architecture with DenseNet-inspired bottleneck and PatchGAN discriminator.

## 📋 Overview

This project implements a robust GAN framework for medical image synthesis:
- **Generator**: U-Net with DenseNet bottleneck (8 dense blocks)
- **Discriminator**: PatchGAN with Wasserstein loss and gradient penalty
- **Loss Functions**: Adversarial + L1 + SSIM for realistic reconstruction
- **Training**: Google Colab-ready with automatic checkpointing

## ✨ Key Features

- ✅ **Complete End-to-End Pipeline**: From data loading to visualization
- ✅ **Robust Training**: WGAN-GP for stable convergence
- ✅ **Multi-Format Support**: PNG, JPG, NIfTI (.nii), DICOM (.dcm)
- ✅ **Automatic Checkpointing**: Save progress to Google Drive
- ✅ **Real-time Monitoring**: CSV metrics logging
- ✅ **Comprehensive Visualization**: Training curves and result comparison

## 🏗️ Architecture

### Generator (UnetDenseNet)
```
Input (1×256×256) 
  ↓
Encoder: 64 → 128 → 256 → 512
  ↓
Bottleneck: 8 Dense Blocks (512 → 1024 → 512)
  ↓
Decoder: 512 → 256 → 128 → 64
  ↓
Output (1×256×256) with Tanh
```

### Discriminator (PatchGAN)
```
Input (2×256×256) [MRI + PET concatenated]
  ↓
C64 → C128 → C256 → C512 → C1
  ↓
Patch-wise predictions
```

## 📁 Project Structure

```
Unet-DenseNet-GAN-MRI2PET/
├── networks.py              # Generator & Discriminator architectures
├── losses.py                # L1, SSIM, and WGAN-GP gradient penalty
├── dataset.py               # Data loading for MRI-PET pairs
├── train.py                 # Main training script
├── analyze_results.ipynb    # Visualization notebook
├── requirements.txt         # Python dependencies
├── QUICK_START.md          # Step-by-step Colab guide
└── WORKFLOW_GUIDE.md       # Complete workflow documentation
```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/Erfanhabibi/Unet-DenseNet-GAN-MRI2PET.git
cd Unet-DenseNet-GAN-MRI2PET
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset
Organize your data as:
```
MRI_PET_Data/
├── train/
│   ├── mri/
│   └── pet/
├── val/
│   ├── mri/
│   └── pet/
└── test/
    ├── mri/
    └── pet/
```

### 4. Train Model
```bash
python train.py \
  --epochs 100 \
  --batch_size 8 \
  --lr 0.0002 \
  --data_path /path/to/MRI_PET_Data \
  --checkpoint_dir /path/to/checkpoints
```

## 💻 Google Colab Training

Perfect for training without local GPU! See [QUICK_START.md](QUICK_START.md) for detailed instructions.

**One-command setup:**
```python
!git clone https://github.com/Erfanhabibi/Unet-DenseNet-GAN-MRI2PET.git
%cd Unet-DenseNet-GAN-MRI2PET
!pip install -q -r requirements.txt
```

## 📊 Dataset Recommendations

- **ADNI**: https://adni.loni.usc.edu/ (Brain MRI + PET)
- **TCIA**: https://www.cancerimagingarchive.net/ (Cancer imaging)
- **IXI Dataset**: https://brain-development.org/ixi-dataset/ (Brain MRI)
- **OASIS**: https://www.oasis-brains.org/ (Brain imaging)

See [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) for complete dataset preparation instructions.

## 📈 Training Details

### Loss Function
```
Total Generator Loss = Adversarial Loss + 100×L1 Loss + 10×SSIM Loss
Total Discriminator Loss = WGAN Loss + 10×Gradient Penalty
```

### Hyperparameters
- **Optimizer**: Adam (β₁=0.5, β₂=0.999)
- **Learning Rate**: 0.0002
- **Batch Size**: 4-8 (depends on GPU)
- **Image Size**: 256×256
- **Epochs**: 100-200 recommended

### Expected Training Time
- **Small dataset (100 images)**: ~2-5 min/epoch (T4 GPU)
- **Medium dataset (1000 images)**: ~20-30 min/epoch (T4 GPU)
- **Large dataset (5000+ images)**: ~1-2 hours/epoch (T4 GPU)

## 📊 Results Visualization

Use the provided Jupyter notebook:
```python
# In analyze_results.ipynb
from networks import UnetDenseNet
import torch

# Load trained model
generator = UnetDenseNet().to('cuda')
checkpoint = torch.load('checkpoints/final_checkpoint.pth')
generator.load_state_dict(checkpoint['generator_state_dict'])

# Generate predictions
# See notebook for complete visualization code
```

## 🛠️ Advanced Usage

### Resume Training from Checkpoint
```python
checkpoint = torch.load('checkpoints/latest_checkpoint.pth')
generator.load_state_dict(checkpoint['generator_state_dict'])
discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
```

### Adjust Loss Weights
Edit `train.py`:
```python
lambda_l1 = 100.0    # L1 loss weight
lambda_ssim = 10.0   # SSIM loss weight
lambda_gp = 10.0     # Gradient penalty weight
```

### Use Custom Transforms
```python
from dataset import MRIPETDataset
import torchvision.transforms as transforms

custom_transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Larger images
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = MRIPETDataset(data_path, transform=custom_transform)
```

## 📚 Documentation

- [QUICK_START.md](QUICK_START.md) - Step-by-step Colab guide
- [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) - Complete workflow from data to results
- [analyze_results.ipynb](analyze_results.ipynb) - Visualization notebook

## 🔧 Troubleshooting

**CUDA out of memory?**
```bash
python train.py --batch_size 2 ...
```

**Training unstable?**
```bash
python train.py --lr 0.0001 ...
```

**Need faster training?**
- Use mixed precision (add to train.py)
- Increase batch size if GPU allows
- Use smaller image size (128×128)

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@misc{unet-densenet-gan-mri2pet,
  author = {Erfanhabibi},
  title = {U-Net/DenseNet GAN for MRI-to-PET Synthesis},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Erfanhabibi/Unet-DenseNet-GAN-MRI2PET}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- U-Net architecture: [Ronneberger et al.](https://arxiv.org/abs/1505.04597)
- DenseNet: [Huang et al.](https://arxiv.org/abs/1608.06993)
- PatchGAN: [Isola et al.](https://arxiv.org/abs/1611.07004)
- WGAN-GP: [Gulrajani et al.](https://arxiv.org/abs/1704.00028)

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**⭐ If you find this project helpful, please consider giving it a star!**