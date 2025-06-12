# fetal_ultrasound_project/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pandas as pd
import os
from torchvision import transforms # Make sure to import transforms
import matplotlib.pyplot as plt # For plotting training graphs

from model.custom_cnn import FetalNet
from utils.dataset import FetalDataset, get_train_transforms, get_val_test_transforms
from utils.train_utils import train_model, validate_model, evaluate_model # Your training utility functions
from config.config import * # Import all constants from config

def main():
    # Set device
    device = torch.device(DEVICE)
    print(f"Using device: {device}")

    # Load data
    df = pd.read_csv(DATA_CSV_PATH)

    # Define transforms
    train_transform = get_train_transforms(IMAGE_SIZE, NORM_MEAN, NORM_STD)
    val_test_transform = get_val_test_transforms(IMAGE_SIZE, NORM_MEAN, NORM_STD)

    # Create full dataset
    full_dataset = FetalDataset(df, IMAGE_DIR, transform=None) # No transform initially for splitting

    # Split data into train, validation, and test sets
    train_size = int(0.70 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # Apply transforms to subsets AFTER splitting
    # This ensures transforms are applied to the correct part of the data.
    # Note: random_split returns Subset objects, which wrap the original dataset.
    # We need to access the underlying dataset to change its transform.
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.transform = val_test_transform

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Initialize model, criterion, optimizer
    model = FetalNet(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
            print(f"Saved best model with Val Acc: {best_val_acc:.4f}")

    print("Training complete.")

    # Load the best model for final evaluation
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    test_acc, all_preds, all_labels = evaluate_model(model, test_loader, device)
    print(f"Final Test Accuracy: {test_acc:.4f}")

    # Optional: Plotting training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Ensure torch is imported globally for DEVICE in config.py
    import torch 
    main()