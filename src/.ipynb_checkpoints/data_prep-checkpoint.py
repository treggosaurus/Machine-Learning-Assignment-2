import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.ffill(inplace=True)
    
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['hour'] = df['Datetime'].dt.hour
    df['day_of_week'] = df['Datetime'].dt.dayofweek
    df['month'] = df['Datetime'].dt.month
    
    df['lag_24h'] = df['PJME_MW'].shift(24)
    df['lag_1week'] = df['PJME_MW'].shift(168)
    
    df['rolling_3h_max'] = df['PJME_MW'].shift(1).rolling(window=3).max()
    df['rolling_24h_max'] = df['PJME_MW'].shift(1).rolling(window=24).max()
    
    df.dropna(inplace=True)
    
    return df