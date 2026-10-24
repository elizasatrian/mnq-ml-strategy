import pandas as pd
import ta_utils

def ml_features_mnq_with_triple_barrier(mnq_hr, target_col='Close', lags=12, atr_mult=1.5, max_holding_bars=6):
    df_ml = mnq_hr.copy()
    
    if 'Hour' not in df_ml.columns:
        df_ml['Hour'] = df_ml.index.hour

    df_ml['return_1'] = df_ml[target_col].pct_change(1)
    df_ml['return_3'] = df_ml[target_col].pct_change(3)
    df_ml['return_6'] = df_ml[target_col].pct_change(6)
    df_ml['return_12'] = df_ml[target_col].pct_change(12)
    df_ml['return_24'] = df_ml[target_col].pct_change(24)

    df_ml['SMA_20'] = ta_utils.sma(df_ml[target_col], length=20)
    df_ml['SMA_50'] = ta_utils.sma(df_ml[target_col], length=50)
    df_ml['EMA_20'] = ta_utils.ema(df_ml[target_col], length=20)
    df_ml['EMA_50'] = ta_utils.ema(df_ml[target_col], length=50)

    df_ml['close_SMA20'] = df_ml[target_col] / df_ml['SMA_20'] - 1
    df_ml['close_SMA50'] = df_ml[target_col] / df_ml['SMA_50'] - 1
    df_ml['close_EMA20'] = df_ml[target_col] / df_ml['EMA_20'] - 1
    df_ml['close_EMA50'] = df_ml[target_col] / df_ml['EMA_50'] - 1

    df_ml['RSI_7'] = ta_utils.rsi(df_ml[target_col], length=7)
    df_ml['RSI_14'] = ta_utils.rsi(df_ml[target_col], length=14)
    df_ml['RSI_28'] = ta_utils.rsi(df_ml[target_col], length=28)

    df_ml['ATR'] = ta_utils.atr(df_ml['High'], df_ml['Low'], df_ml['Close'], length=14)
    df_ml['volatility_regime'] = (df_ml['ATR'] > df_ml['ATR'].rolling(50).mean()).astype(int)

    macd = ta_utils.macd(df_ml[target_col])
    if macd is not None and not macd.empty:
        df_ml['MACD'] = macd.iloc[:, 0]
        df_ml['MACD_signal'] = macd.iloc[:, 1]
        df_ml['MACD_hist'] = macd.iloc[:, 2]

    df_ml['volatility_6'] = df_ml['return_1'].rolling(6).std()
    df_ml['volatility_12'] = df_ml['return_1'].rolling(12).std()
    df_ml['volatility_24'] = df_ml['return_1'].rolling(24).std()

    df_ml['range'] = (df_ml['High'] - df_ml['Low']) / df_ml[target_col]
    df_ml['open_close'] = (df_ml['Close'] - df_ml['Open']) / df_ml['Open']
    df_ml['relative_volume'] = df_ml['Volume'] / df_ml['Volume'].rolling(20).mean()

    lag_columns = ['return_1', 'return_3', 'RSI_14', 'MACD_hist', 'volatility_12', 'relative_volume']
    lag_features = {}
    for col in lag_columns:
        if col in df_ml.columns:
            for lag in range(1, lags + 1):
                lag_features[f'{col}_lag{lag}'] = df_ml[col].shift(lag)

    df_ml = pd.concat([df_ml, pd.DataFrame(lag_features, index=df_ml.index)], axis=1)

    return df_ml

def compute_features(df):
    """Wrapper function to compute features from raw OHLCV data"""
    return ml_features_mnq_with_triple_barrier(df)
