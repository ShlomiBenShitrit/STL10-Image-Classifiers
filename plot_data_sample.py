import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import config
import get_data
import os

def plot_task1_class_grid():
    """
    Task 1: Plot 4 examples from each of the 10 classes in a 10x4 grid.
    Requirement: Each row is labeled with the class name.
    """
    # 1. Ensure the output directory exists
    output_folder = 'plots_and_outputs'
    os.makedirs(output_folder, exist_ok=True)

    print("[Info] Generating Task 1: 10x4 Class Grid...")
    dataset = datasets.STL10(root=config.DATA_ROOT, split='train', download=False)
    classes = dataset.classes
    
    fig, axes = plt.subplots(10, 4, figsize=(12, 20))
    fig.suptitle('STL-10: 4 Samples per Class (Task 1)', fontsize=16)

    for i, class_name in enumerate(classes):
        # Filter indices for each specific class
        indices = [idx for idx, label in enumerate(dataset.labels) if label == i][:4]
        
        # Label each row with the class name
        axes[i, 0].set_ylabel(class_name, rotation=0, labelpad=45, fontsize=12, fontweight='bold')
        
        for j, idx in enumerate(indices):
            img, _ = dataset[idx] # Raw 96x96 PIL image
            axes[i, j].imshow(img)
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save to the specific folder
    save_path = os.path.join(output_folder, 'task1_class_grid.png')
    plt.savefig(save_path)
    print(f"[Info] Saved: {save_path}")
    plt.close()

def plot_task2_individual_augmentations():
    """
    Task 2: Show each individual augmentation to size 64 as per instructions.
    """
    # 1. Ensure the output directory exists
    output_folder = 'plots_and_outputs'
    os.makedirs(output_folder, exist_ok=True)

    print("[Info] Generating Task 2: Individual Augmentation Samples...")
    dataset = datasets.STL10(root=config.DATA_ROOT, split='train', download=False)
    img_raw, label = dataset[0] # Original 96x96 image
    
    # Defining the specific augmentations used in train_transform
    augmentations = [
        ("Original (96x96)", None),
        ("Center Crop (64x64)", transforms.CenterCrop(64)),
        ("Random Crop (64x64)", transforms.RandomCrop(64)),
        ("Horizontal Flip", transforms.RandomHorizontalFlip(p=1.0)),
        ("Random Rotation (15°)", transforms.RandomRotation(degrees=(15, 15))),
        ("Color Jitter", transforms.ColorJitter(brightness=0.5))
    ]
    
    fig, axes = plt.subplots(1, len(augmentations), figsize=(22, 5))
    fig.suptitle('Breakdown of Direct 64x64 Cropping and Augmentations (Task 2)', fontsize=16)

    for i, (name, aug) in enumerate(augmentations):
        # Apply augmentation directly to the raw 96x96 image
        display_img = img_raw if aug is None else aug(img_raw)
        
        axes[i].imshow(display_img)
        axes[i].set_title(name, fontsize=11)
        axes[i].axis('off')

    plt.tight_layout()
    
    # Save to the specific folder
    save_path = os.path.join(output_folder, 'task2_individual_augmentations.png')
    plt.savefig(save_path)
    print(f"[Info] Saved: {save_path}")
    plt.close()

if __name__ == "__main__":
    plot_task1_class_grid()
    plot_task2_individual_augmentations()
    print("\n[Success] Visualization complete!")