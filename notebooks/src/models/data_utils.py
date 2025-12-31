
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class MultiAssetDataset(Dataset):
    """
    PyTorch Dataset for multi-asset prediction
    Maps 3-class labels {-1, 0, 1} to {0, 1, 2} for CrossEntropyLoss
    """
    
    def __init__(self, features_df, targets_df, symbol=None):
        """
        Args:
            features_df: DataFrame with features
            targets_df: DataFrame with targets (one column per symbol)
            symbol: Which symbol to predict (if None, use first column)
        """
        # Align indices
        common_idx = features_df.index.intersection(targets_df.index)
        self.features = features_df.loc[common_idx].values.astype(np.float32)
        
        # Select target symbol
        if symbol is None:
            symbol = targets_df.columns[0]
        self.targets = targets_df.loc[common_idx, symbol].values
        
        # Map {-1, 0, 1} to {0, 1, 2} for PyTorch
        self.targets = (self.targets + 1).astype(np.int64)
        
        self.symbol = symbol
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (torch.from_numpy(self.features[idx]),
                torch.tensor(self.targets[idx]))


def create_walk_forward_splits(features_df, targets_df, 
                                train_size=25000, test_size=12500,
                                n_splits=10, step=1000):
    """
    Create walk-forward validation splits following Dixon methodology
    
    Args:
        features_df: Feature matrix
        targets_df: Target labels
        train_size: Training window size (Dixon: 25,000)
        test_size: Test window size (Dixon: 12,500)
        n_splits: Number of folds (Dixon: 10)
        step: Step size between folds (Dixon: 1,000)
    
    Returns:
        List of (train_features, train_targets, test_features, test_targets)
    """
    splits = []
    
    for i in range(n_splits):
        start_idx = i * step
        train_end = start_idx + train_size
        test_end = train_end + test_size
        
        if test_end > len(features_df):
            break
        
        train_feat = features_df.iloc[start_idx:train_end]
        train_targ = targets_df.iloc[start_idx:train_end]
        
        test_feat = features_df.iloc[train_end:test_end]
        test_targ = targets_df.iloc[train_end:test_end]
        
        splits.append((train_feat, train_targ, test_feat, test_targ))
    
    print(f"Created {len(splits)} walk-forward splits")
    print(f"Train size: {train_size}, Test size: {test_size}")
    
    return splits


def prepare_dataloaders(train_features, train_targets, 
                        test_features, test_targets,
                        symbol, batch_size=256):
    """
    Create PyTorch DataLoaders for training and testing
    
    Args:
        batch_size: Mini-batch size (Dixon uses batching for efficiency)
    """
    # Create datasets
    train_dataset = MultiAssetDataset(train_features, train_targets, symbol)
    test_dataset = MultiAssetDataset(test_features, test_targets, symbol)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,  # Shuffle for SGD
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, test_loader
