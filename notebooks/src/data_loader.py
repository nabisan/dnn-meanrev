
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from tqdm import tqdm

class MultiAssetDataLoader:
    """
    Load 5-minute OHLCV data for multiple instruments
    Following Dixon et al. 2017 methodology
    """
    
    def __init__(self, symbols, start_date, end_date, data_source='crypto'):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.data_source = data_source
        self.data = {}
        
    def download_crypto_data(self):
        """Download crypto data using CCXT (5-min candles)"""
        exchange = ccxt.binance()
        
        for symbol in tqdm(self.symbols, desc="Downloading crypto data"):
            try:
                # CCXT uses milliseconds
                since = int(pd.Timestamp(self.start_date).timestamp() * 1000)
                
                all_candles = []
                current_since = since
                
                while True:
                    candles = exchange.fetch_ohlcv(
                        symbol, 
                        timeframe='5m',
                        since=current_since,
                        limit=1000
                    )
                    
                    if not candles or len(candles) == 0:
                        break
                        
                    all_candles.extend(candles)
                    current_since = candles[-1][0] + 1
                    
                    # Check if we've reached end date
                    if candles[-1][0] >= int(pd.Timestamp(self.end_date).timestamp() * 1000):
                        break
                
                # Convert to DataFrame
                df = pd.DataFrame(
                    all_candles,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                self.data[symbol] = df
                print(f"✓ {symbol}: {len(df)} candles")
                
            except Exception as e:
                print(f"✗ Error downloading {symbol}: {e}")
                
        return self.data
    
    def load_data(self):
        """Main method to load data based on source"""
        if self.data_source == 'crypto':
            return self.download_crypto_data()
        else:
            raise NotImplementedError("Only crypto supported for now")
    
    def save_data(self, output_dir='data/raw'):
        """Save downloaded data to CSV"""
        for symbol, df in self.data.items():
            filename = f"{output_dir}/{symbol.replace('/', '_')}.csv"
            df.to_csv(filename)
            print(f"Saved: {filename}")
