import pandas as pd
from typing import List, Tuple

class WalkForwardEngine:
    """Splits data into rolling walk-forward windows."""
    
    def __init__(self, df: pd.DataFrame, train_years: int = 3, test_years: int = 1):
        self.df = df
        self.train_years = train_years
        self.test_years = test_years
        if not pd.api.types.is_datetime64_any_dtype(self.df.index):
            self.df.index = pd.to_datetime(self.df.index)
            
    def get_windows(self) -> List[Tuple[pd.DataFrame, pd.DataFrame, str, str]]:
        windows = []
        start_year = self.df.index.year.min()
        max_year = self.df.index.year.max()
        
        current_year = start_year
        while current_year + self.train_years <= max_year:
            train_start = f"{current_year}-01-01"
            train_end = f"{current_year + self.train_years - 1}-12-31 23:59:59"
            test_start = f"{current_year + self.train_years}-01-01"
            test_end = f"{current_year + self.train_years + self.test_years - 1}-12-31 23:59:59"
            
            df_train = self.df.loc[train_start:train_end]
            df_test = self.df.loc[test_start:test_end]
            
            if not df_test.empty and not df_train.empty:
                windows.append((df_train, df_test, f"{current_year}-{current_year + self.train_years - 1}", f"{current_year + self.train_years}"))
                
            current_year += 1
            
        return windows
