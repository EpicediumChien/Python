import pandas as pd
import numpy as np
import os
from scipy import signal

# 設定檔案路徑
path = r'C:\Git\Python\IMU\data\AccelerationVariance' 

def apply_agc_logic(sig, fs=20, window_sec=10):
    window_size = int(fs * window_sec)
    # 使用 rolling 計算滾動標準差
    rolling_std = pd.Series(sig).rolling(window=window_size, center=True).std()
    # 填充缺失值並避免除以零
    rolling_std = rolling_std.bfill().ffill().replace(0, 1e-6)
    return sig / rolling_std.values

def clean_signal_logic(data, fs=20):
    """
    預處理邏輯：
    1. 2階帶通濾波 [0.08, 0.85] Hz
    2. AGC 自動增益補償 (10秒視窗)
    3. Savitzky-Golay 平滑濾波器
    """
    cleaned_data = np.zeros_like(data)
    nyq = 0.5 * fs
    # 建立 2階 帶通濾波器
    b, a = signal.butter(2, [0.08/nyq, 0.85/nyq], btype='band')
    
    # 對每一列 (X, Y, Z) 分別處理
    for i in range(data.shape[1]):
        # A. 帶通濾波
        feat_filt = signal.filtfilt(b, a, data[:, i])
        
        # B. AGC 處理 (標準化振幅)
        feat_agc = apply_agc_logic(feat_filt, fs=fs, window_sec=10)
        
        # C. Savitzky-Golay 平滑
        try:
            # window_length=15, polyorder=2
            cleaned_data[:, i] = signal.savgol_filter(feat_agc, 15, 2)
        except:
            cleaned_data[:, i] = feat_agc
            
    return cleaned_data

def calculate_processed_variance(file_name, fs=20):
    file_path = os.path.join(path, file_name)
    
    try:
        # 讀取 CSV
        df = pd.read_csv(file_path)
        
        # 檢查欄位
        acc_cols = ['AccX', 'AccY', 'AccZ'] 
        if not all(col in df.columns for col in acc_cols):
            return f"找不到欄位 {acc_cols}"

        # 提取數據轉為 numpy array (rows, cols)
        raw_data = df[acc_cols].values
        
        # --- 執行濾波預處理 ---
        # 這裡會對 X, Y, Z 分別進行帶通、AGC、平滑
        cleaned_data = clean_signal_logic(raw_data, fs=fs)
        
        # --- 計算處理後的合加速度 Magnitude ---
        # 處理後的訊號已經去除了直流偏置(DC offset)，因此平方和再開根號能反應震盪強度
        acc_mag = np.sqrt(cleaned_data[:, 0]**2 + cleaned_data[:, 1]**2 + cleaned_data[:, 2]**2)
        
        # 計算方差 (Variance)
        # 經過 AGC 後的方差會反映訊號的「乾淨度」與「穩定度」
        variance = np.var(acc_mag)
        
        return variance

    except Exception as e:
        return f"處理失敗: {e}"

# 執行驗證
files = ["30 BPM.csv", "30 BPM - Straw.csv"]

print(f"--- 經過濾波預處理之加速度方差 (fs=20) ---")
for f in files:
    var_result = calculate_processed_variance(f, fs=20)
    if isinstance(var_result, float):
        print(f"檔案: {f:25} | 預處理後方差: {var_result:.6f}")
    else:
        print(f"檔案: {f:25} | 錯誤: {var_result}")