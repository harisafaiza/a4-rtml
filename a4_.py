# -*- coding: utf-8 -*-
"""A4..ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1JhFXIpHDqmi_BmePnz0Lxd0dk5CH72FT
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define MAE Encoder
class MAE_Encoder(nn.Module):
    def __init__(self, image_size, patch_size, emb_dim, num_layer, num_head, mask_ratio):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.num_layer = num_layer
        self.num_head = num_head
        self.mask_ratio = mask_ratio

        self.patchify = nn.Conv2d(1, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_head), num_layers=num_layer
        )
        self.decoder = nn.ConvTranspose2d(emb_dim, 1, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        patches = self.patchify(x)
        patches = patches.flatten(2).transpose(1, 2)  # (B, N, C)
        encoded = self.transformer(patches)
        decoded = encoded.transpose(1, 2).view(x.shape[0], self.emb_dim, self.image_size // self.patch_size, self.image_size // self.patch_size)
        return self.decoder(decoded), encoded

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Define MAE training function
def train_mae(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, _ in tqdm(dataloader):
            images = images.to(device)
            optimizer.zero_grad()
            reconstructed, _ = model(images)
            loss = criterion(reconstructed, images)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

# Define MAE evaluation function
def evaluate_mae(model, dataloader):
    model.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            reconstructed, _ = model(images)
            break  # Only visualize first batch
    return images, reconstructed

# Train the classifier
def train_classifier(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss, correct = 0, 0
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
        accuracy = 100 * correct / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%")

# Main script execution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize MAE model
mae_model = MAE_Encoder(image_size=28, patch_size=4, emb_dim=128, num_layer=6, num_head=4, mask_ratio=0.75).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(mae_model.parameters(), lr=1e-3)

# Train MAE on MNIST
train_mae(mae_model, train_loader, criterion, optimizer, num_epochs=10)

# Evaluate reconstruction
originals, reconstructions = evaluate_mae(mae_model, test_loader)

# Visualize results
fig, axes = plt.subplots(2, 10, figsize=(10, 2))
for i in range(10):
    axes[0, i].imshow(originals[i].cpu().squeeze(), cmap='gray')
    axes[1, i].imshow(reconstructions[i].cpu().squeeze(), cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].axis('off')
plt.show()

# Fine-tune the encoder for classification
classifier = ViT_Classifier(mae_model, num_classes=10).to(device)
criterion_cls = nn.CrossEntropyLoss()
optimizer_cls = optim.Adam(classifier.parameters(), lr=1e-3)
train_classifier(classifier, train_loader, criterion_cls, optimizer_cls, num_epochs=10)

# Load CIFAR-10 dataset
transform_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
cifar_train = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform_cifar, download=True)
cifar_test = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform_cifar, download=True)
cifar_train_loader = DataLoader(cifar_train, batch_size=128, shuffle=True)
cifar_test_loader = DataLoader(cifar_test, batch_size=128, shuffle=False)

# Train MAE on CIFAR-10
mae_model_cifar = MAE_Encoder(image_size=32, patch_size=4, emb_dim=128, num_layer=6, num_head=4, mask_ratio=0.75).to(device)
optimizer_cifar = optim.Adam(mae_model_cifar.parameters(), lr=1e-3)
train_mae(mae_model_cifar, cifar_train_loader, criterion, optimizer_cifar, num_epochs=10)

# Define ViT Classifier using the trained encoder
class ViT_Classifier(nn.Module):
    def __init__(self, encoder: MAE_Encoder, num_classes=10):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, encoder.emb_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, (encoder.image_size // encoder.patch_size) ** 2 + 1, encoder.emb_dim))
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.decoder
        self.head = nn.Linear(encoder.emb_dim, num_classes)

    def forward(self, img):
        patches = self.patchify(img)
        patches = patches.flatten(2).transpose(1, 2)  # (B, N, C)
        batch_size, num_patches, _ = patches.shape

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        patches = torch.cat([cls_tokens, patches], dim=1)
        patches = patches + self.pos_embedding[:, :num_patches + 1, :]

        features = self.layer_norm(self.transformer(patches))
        logits = self.head(features[:, 0, :])  # Extract classification token output
        return logits

# Fine-tune classifier on CIFAR-10
classifier_cifar = ViT_Classifier(mae_model_cifar, num_classes=10).to(device)
optimizer_cls_cifar = optim.Adam(classifier_cifar.parameters(), lr=1e-3)
train_classifier(classifier_cifar, cifar_train_loader, criterion_cls, optimizer_cls_cifar, num_epochs=10)