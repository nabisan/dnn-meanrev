
import pandas as pd
import numpy as np
from scipy.stats import zscore

class DixonFeatureEngineer:
    """
    Feature engineering following Dixon et al. 2017
    - Lagged returns (1-100)
    - Moving averages (5-100)
    - Cross-instrument correlations
    """
    
    def __init__(self, data_dict, normalize=True):
        """
        Args:
            data_dict: {symbol: DataFrame} with OHLCV data
            normalize: Apply z-score normalization (Dixon method)
        """
        self.data_dict = data_dict
        self.normalize = normalize
        self.features_df = None
        
    def compute_returns(self):
        """Compute log returns for all symbols"""
        returns_dict = {}
        
        for symbol, df in self.data_dict.items():
            returns = np.log(df['close'] / df['close'].shift(1))
            returns_dict[symbol] = returns.dropna()
            
        return returns_dict
    
    def create_lag_features(self, returns, symbol, max_lag=100):
        """
        Create lagged return features (t-1 to t-max_lag)
        Dixon uses lags 1-100
        """
        features = pd.DataFrame(index=returns.index)
        
        for lag in range(1, max_lag + 1):
            features[f'{symbol}_lag_{lag}'] = returns.shift(lag)
            
        return features
    
    def create_ma_features(self, returns, symbol, windows=[5, 10, 20, 50, 100]):
        """
        Create moving average features
        Dixon uses windows 5-100
        """
        features = pd.DataFrame(index=returns.index)
        
        for window in windows:
            features[f'{symbol}_ma_{window}'] = returns.rolling(window).mean()
            
        return features
    
    def create_correlation_features(self, returns_dict, windows=[20, 50, 100]):
        """
        Create pairwise correlation features between all symbols
        This is KEY innovation in Dixon paper - capturing co-movements
        """
        symbols = list(returns_dict.keys())
        features = pd.DataFrame()
        
        # Align all returns to same index
        aligned_returns = pd.DataFrame({
            symbol: returns_dict[symbol] 
            for symbol in symbols
        }).dropna()
        
        for window in windows:
            # Rolling correlation for each pair
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    sym1, sym2 = symbols[i], symbols[j]
                    
                    corr = aligned_returns[sym1].rolling(window).corr(
                        aligned_returns[sym2]
                    )
                    
                    col_name = f'corr_{sym1}_{sym2}_w{window}'
                    features[col_name] = corr
                    
        features.index = aligned_returns.index
        return features
    
    def create_all_features(self):
        """Generate complete feature matrix"""
        print("Computing returns...")
        returns_dict = self.compute_returns()
        
        all_features = []
        
        # Per-symbol features (lags + MAs)
        for symbol in self.data_dict.keys():
            print(f"Creating features for {symbol}...")
            
            returns = returns_dict[symbol]
            
            # Lags
            lag_feats = self.create_lag_features(returns, symbol, max_lag=100)
            all_features.append(lag_feats)
            
            # Moving averages
            ma_feats = self.create_ma_features(returns, symbol)
            all_features.append(ma_feats)
        
        # Cross-instrument correlations
        print("Computing cross-instrument correlations...")
        corr_feats = self.create_correlation_features(returns_dict)
        all_features.append(corr_feats)
        
        # Combine all features
        print("Combining features...")
        self.features_df = pd.concat(all_features, axis=1)
        
        # Normalize (Dixon: subtract mean, divide by std)
        if self.normalize:
            print("Normalizing features...")
            self.features_df = self.features_df.apply(zscore, nan_policy='omit')
        
        # Drop NaN rows
        self.features_df = self.features_df.dropna()
        
        print(f"✓ Feature matrix shape: {self.features_df.shape}")
        print(f"  Total features: {self.features_df.shape[1]}")
        print(f"  Total observations: {self.features_df.shape[0]}")
        
        return self.features_df
    
    def save_features(self, output_path='data/processed/features.csv'):
        """Save feature matrix"""
        self.features_df.to_csv(output_path)
        print(f"Features saved to {output_path}")
