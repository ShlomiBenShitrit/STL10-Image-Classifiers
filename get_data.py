import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import config  # Imported to access IMAGE_SIZE, BATCH_SIZE, DATA_ROOT, etc.

# ==================================================================================
# CLASS: DatasetWithTransform
# Description: A wrapper class used within this script to apply different 
#              transformations to different subsets (Train/Val) of the same dataset.
# Used in: This script (get_data_loaders function).
# ==================================================================================
class DatasetWithTransform(Dataset):
    """
    Wraps a dataset subset to apply specific transformations.
    
    Args:
        subset (torch.utils.data.Subset): A subset of the original STL-10 dataset.
        transform (callable, optional): The transformation pipeline to apply to images.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset      # Stores the list of indices for this split
        self.transform = transform # Stores the specific augmentation/processing rules

    def __getitem__(self, index):
        """
        Fetched automatically by the DataLoader during the training loop in 'train.py'.
        
        Args:
            index (int): The index of the sample to retrieve.
            
        Returns:
            tuple: (image, label) where image is a processed Tensor and label is an int.
        """
        # 1. Retrieve the raw image and label from the original dataset
        x, y = self.subset[index]
        
        # 2. Apply the specific transform (Augmentation for train, Clean for val/test)
        if self.transform:
            x = self.transform(x)
            
        return x, y

    def __len__(self):
        return len(self.subset)


# ==================================================================================
# FUNCTION: get_data_loaders
# Description: Prepares the entire data pipeline. 
# Used in: 'train.py' for training and 'evaluate.py' for final testing.
# Example usage in train.py: train_loader, val_loader, test_loader = get_data_loaders()
# ==================================================================================
def get_data_loaders():
    """
    Downloads STL-10, splits it, and wraps segments into DataLoaders.
    
    Returns:
        tuple: (train_loader, val_loader, test_loader) - objects used to stream data.
    """
    print("[Info] Preparing data transformations...")
    
    # 1. Define Normalization values based on config.MODEL_TYPE
    # MobileNet uses ImageNet stats; simple models use standard 0.5 centering.
    if 'mobilenet' in config.MODEL_TYPE:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    # 2. Training Transforms: Includes random augmentations to prevent overfitting.
    # These are triggered every time a batch is requested in 'train.py'.
    train_transform = transforms.Compose([
        transforms.RandomCrop(config.IMAGE_SIZE), # Requirement: Random crop to 64x64        
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomRotation(degrees=15),  
        transforms.ColorJitter(brightness=0.2), 
        transforms.ToTensor(), # Convert back to PyTorch Tensor
        transforms.Normalize(mean=mean, std=std) # Normalize pixel values
    ])

    # 3. Evaluation Transforms: Used for Validation and Testing.
    # No random augmentations here; only fixed resizing/cropping for consistency.
    eval_transform = transforms.Compose([
        transforms.CenterCrop(config.IMAGE_SIZE), # Fixed center crop to target size
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # 4. Load Raw Data from disk (or download if missing)
    # Uses DATA_ROOT path defined in 'config.py'.
    full_train_raw = datasets.STL10(root=config.DATA_ROOT, split='train', download=True)
    test_raw = datasets.STL10(root=config.DATA_ROOT, split='test', download=True)

    # 5. Split Training set into Train and Validation (80/20 ratio)
    # Ratios and Seeds are pulled from 'config.py' for reproducibility.
    num_val = int(len(full_train_raw) * config.VALIDATION_SPLIT)
    num_train = len(full_train_raw) - num_val
    
    train_subset, val_subset = random_split(
        full_train_raw, [num_train, num_val], 
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )

    # 6. Instantiate DatasetWithTransform objects.
    # This triggers the __init__ method of the class above.
    train_dataset = DatasetWithTransform(train_subset, transform=train_transform)
    val_dataset = DatasetWithTransform(val_subset, transform=eval_transform)
    test_dataset = DatasetWithTransform(test_raw, transform=eval_transform)

    # 7. Create DataLoaders (The "pumps" that feed data to the model)
    # Batch size is fetched dynamically from config.CURRENT_PARAMS.
    batch_size = config.CURRENT_PARAMS['batch_size']
    
    # train_loader: Used in 'train.py' main loop; shuffle=True for better learning.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=2, pin_memory=True)
    
    # val_loader: Used in 'train.py' at the end of each epoch to check performance.
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=2, pin_memory=True)
    
    # test_loader: Used in 'evaluate.py' for the final model assessment.
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=2, pin_memory=True)

    print(f"[Info] DataLoaders ready. Training on: {num_train} samples.")
    return train_loader, val_loader, test_loader

# ==================================================================================
# SELF-TEST BLOCK
# Runs only when this script is executed directly.
# ==================================================================================
if __name__ == "__main__":
    t_loader, v_loader, _ = get_data_loaders()
    # Fetch one batch to verify the output shape matches model requirements
    imgs, _ = next(iter(t_loader))
    print(f"Batch image shape: {imgs.shape}") # Expected: [Batch_Size, 3, 64, 64]