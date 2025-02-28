# Masked Autoencoder (MAE) with MNIST and CIFAR-10

## Overview
This project implements a Masked Autoencoder (MAE) using PyTorch to reconstruct images from the MNIST and CIFAR-10 datasets. The trained encoder is then repurposed for downstream classification tasks.

## Features
- Implements a MAE for self-supervised learning on MNIST and CIFAR-10 datasets.
- Uses a Vision Transformer (ViT)-like encoder to learn meaningful representations.
- Fine-tunes the encoder for image classification after pretraining.
- Supports experimentation with different patch sizes and masking ratios.
- Visualizes original and reconstructed images.

## Installation
Ensure you have Python 3.7+ and install the required dependencies:

```sh
pip install torch torchvision matplotlib tqdm
```

## Dataset
The script automatically downloads and processes:
- **MNIST**: 28x28 grayscale handwritten digits.
- **CIFAR-10**: 32x32 RGB images of 10 object classes.

## Model Architecture
- **Encoder**: A Vision Transformer-like model with a patch embedding layer and a transformer encoder.
- **Decoder**: A simple transposed convolutional network to reconstruct the images.
- **Classifier**: A linear classifier added on top of the pretrained encoder.

## Training
1. **Pretraining (Self-supervised Learning)**
   - The autoencoder is trained to reconstruct masked images.
2. **Fine-tuning (Supervised Learning)**
   - The pretrained encoder is used for classification by adding a linear classifier.

### Running the Training
To train the MAE and classifier on MNIST:
```sh
python main.py --dataset MNIST
```
To train on CIFAR-10:
```sh
python main.py --dataset CIFAR-10
```

## Results
- Reconstruction quality is evaluated visually by comparing original and reconstructed images.
- Classification performance is measured in terms of accuracy.

## Observations
- Smaller patch sizes generally lead to better reconstruction but increase computation time.
- Higher masking ratios make the task harder but improve learned representations.
- MNIST benefits more from MAE due to its simpler structure compared to CIFAR-10.

## Future Improvements
- Implementing contrastive learning for better feature representations.
- Experimenting with larger datasets such as ImageNet.
- Optimizing hyperparameters for improved classification accuracy.



