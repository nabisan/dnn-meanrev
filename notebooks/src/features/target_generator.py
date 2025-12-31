
import pandas as pd
import numpy as np

class DixonTargetGenerator:
    """
    Generate 3-class labels {-1, 0, +1} for next-period direction
    Following Dixon methodology: threshold = 1e-3
    """
    
    def __init__(self, data_dict, threshold=1e-3):
        self.data_dict = data_dict
        self.threshold = threshold
        self.targets = None
        
    def create_targets(self, primary_symbol=None):
        """
        Create directional labels for each symbol
        
        Args:
            primary_symbol: If None, creates targets for all symbols
        """
        if primary_symbol:
            symbols = [primary_symbol]
        else:
            symbols = list(self.data_dict.keys())
            
        targets_dict = {}
        
        for symbol in symbols:
            df = self.data_dict[symbol]
            
            # Future return (t+1)
            future_return = np.log(df['close'].shift(-1) / df['close'])
            
            # Apply threshold for 3-class classification
            labels = pd.Series(index=df.index, dtype=int)
            labels[future_return > self.threshold] = 1   # UP
            labels[future_return < -self.threshold] = -1  # DOWN
            labels[abs(future_return) <= self.threshold] = 0  # FLAT
            
            targets_dict[symbol] = labels
            
            # Print class distribution
            class_dist = labels.value_counts(normalize=True)
            print(f"{symbol} class distribution:")
            print(f"  UP (+1):   {class_dist.get(1, 0):.2%}")
            print(f"  FLAT (0):  {class_dist.get(0, 0):.2%}")
            print(f"  DOWN (-1): {class_dist.get(-1, 0):.2%}")
            
        self.targets = pd.DataFrame(targets_dict).dropna()
        return self.targets
    
    def save_targets(self, output_path='data/processed/targets.csv'):
        """Save targets"""
        self.targets.to_csv(output_path)
        print(f"Targets saved to {output_path}")
