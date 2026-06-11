import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow_datasets as tfds
import numpy as np

class TFDSBridgeDataset(Dataset):
    """Bridges a TensorFlow Dataset over to PyTorch format"""
    def __init__(self, tf_dataset, is_training=True, image_size=64):
        # Convert the TF dataset stream into a fast memory-cached iterable
        self.data = list(tfds.as_numpy(tf_dataset))
        self.is_training = is_training
        self.image_size = image_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        
        # 1. Normalize image pixels to [0, 1] and make it channels-first (C, H, W)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.tensor(image)
        
        # 2. Add quick runtime data augmentations if it's the training split
        if self.is_training:
            if torch.rand(1).item() > 0.5:
                image_tensor = torch.flip(image_tensor, dims=[2]) # Horizontal flip
            if torch.rand(1).item() > 0.5:
                image_tensor = torch.flip(image_tensor, dims=[1]) # Vertical flip

        # 3. Apply standard ImageNet normalization values
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        return image_tensor, torch.tensor(label, dtype=torch.long)

def get_data_loaders(config):
    # Stream data using your original TFDS splitting logic
    (ds_train, ds_test), ds_info = tfds.load(
        config['data']['dataset_name'],
        split=['train[:80%]', 'train[80%:]'],
        with_info=True,
        as_supervised=True,
        download_and_prepare_kwargs={
            "download_config": tfds.download.DownloadConfig(
                manual_dir="C:/Users/bowli/tensorflow_datasets/downloads/manual/"
            )
        }
    )
    
    # Extract hardcoded class labels from TFDS metadata
    class_names = ds_info.features['label'].names
    
    # Pack them into PyTorch compatible Dataset wrappers
    train_dataset = TFDSBridgeDataset(ds_train, is_training=True, image_size=config['model']['image_size'])
    val_dataset = TFDSBridgeDataset(ds_test, is_training=False, image_size=config['model']['image_size'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False)
    
    return train_loader, val_loader, class_names