import os
import sys
import json
import zipfile
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy import signal
from collections import deque

# ==========================================
# 1. 終極篡改補丁：在「門口」就攔截配置錯誤
# ==========================================
import tf_keras as keras

# --- A. 徹底篡改 MultiHeadAttention (解決 query_shape KeyError) ---
# 我們不再呼叫原始的 from_config，因為原始代碼會導致 KeyError
def ultimate_mha_from_config(cls, config):
    config = config.copy()
    # 1. 補齊 Keras 2 強制要 pop() 的所有形狀參數
    for req_shape in ["query_shape", "key_shape", "value_shape", "output_shape"]:
        if req_shape not in config:
            config[req_shape] = None
            
    # 2. 移除會讓 Keras 2 建構子崩潰的 Keras 3 垃圾
    garbage = ["build_config", "seed", "attention_axes", "registered_name", "module"]
    for k in garbage:
        config.pop(k, None)
        
    # 3. 模擬 Keras 2 的內部 pop 邏輯，防止報錯
    q_shape = config.pop("query_shape")
    k_shape = config.pop("key_shape")
    v_shape = config.pop("value_shape")
    o_shape = config.pop("output_shape")
    
    # 4. 直接呼叫類別建構子，完全繞過原生的崩潰代碼
    return cls(**config)

# 強行替換整個類別方法
keras.layers.MultiHeadAttention.from_config = classmethod(ultimate_mha_from_config)

# --- B. 徹底篡改 LayerNormalization (解決 rms_scaling) ---
def ultimate_ln_from_config(cls, config):
    config = config.copy()
    config.pop("rms_scaling", None)
    config.pop("build_config", None)
    return cls(**config)

keras.layers.LayerNormalization.from_config = classmethod(ultimate_ln_from_config)

# --- C. 徹底篡改 InputLayer (解決 batch_shape) ---
def ultimate_input_from_config(cls, config):
    config = config.copy()
    if "batch_shape" in config:
        config["batch_input_shape"] = config.pop("batch_shape")
    config.pop("sparse", None)
    config.pop("ragged", None)
    return cls(**config)

keras.layers.InputLayer.from_config = classmethod(ultimate_input_from_config)

# ==========================================
# 2. 配置清理與環境偽裝
# ==========================================

def convert_inbound_nodes(nodes):
    """
    將 Keras 3 的 inbound_nodes 格式轉換為 Keras 2 相容格式。
    
    Keras 3: [{"args": [{"keras_history": ["layer", 0, 0]}], "kwargs": {}}]
    Keras 2: [[["layer", 0, 0, {}]]]
    """
    result = []
    for node in nodes:
        if isinstance(node, dict):
            # Keras 3 格式：取出 args 中的 keras_history
            args = node.get("args", [])
            call_args = []
            for arg in args:
                if isinstance(arg, dict) and "keras_history" in arg:
                    kh = arg["keras_history"]
                    # keras_history: [layer_name, node_index, tensor_index]
                    call_args.append([kh[0], kh[1], kh[2], {}])
                elif isinstance(arg, list):
                    # 可能是多個 tensor 的列表
                    inner = []
                    for item in arg:
                        if isinstance(item, dict) and "keras_history" in item:
                            kh = item["keras_history"]
                            inner.append([kh[0], kh[1], kh[2], {}])
                    call_args.extend(inner) if inner else call_args.append(arg)
            result.append(call_args)
        elif isinstance(node, list):
            # 已經是 Keras 2 格式，直接保留
            result.append(node)
    return result


def clean_config_recursive(obj):
    if isinstance(obj, list):
        return [clean_config_recursive(i) for i in obj]
    elif isinstance(obj, dict):
        if obj.get('class_name') == 'Functional':
            obj['class_name'] = 'Model'
        new_obj = {}
        for k, v in obj.items():
            if k in ['module', 'registered_name', 'frozen', 'shared_object_id', 'build_config']:
                continue
            if k == 'dtype' and isinstance(v, dict):
                new_obj[k] = v.get('config', {}).get('name', 'float32')
            elif k == 'inbound_nodes' and isinstance(v, list):
                # ★ 關鍵修正：轉換 inbound_nodes 格式
                new_obj[k] = convert_inbound_nodes(v)
            else:
                new_obj[k] = clean_config_recursive(v)
        return new_obj
    else:
        return obj

def load_keras3_model_compat(model_path, custom_objects=None):
    with zipfile.ZipFile(model_path, 'r') as zip_ref:
        with zip_ref.open('config.json') as f:
            config_data = json.load(f)
        cleaned_config = clean_config_recursive(config_data)

        # 使用公開 API，不依賴不穩定的內部路徑
        import tf_keras as keras_pub
        model = keras_pub.models.model_from_json(
            json.dumps(cleaned_config),
            custom_objects=custom_objects
        )

        weights_path = "model.weights.h5"
        if weights_path in zip_ref.namelist():
            zip_ref.extract(weights_path, ".")
            model.load_weights(weights_path)
            os.remove(weights_path)
        return model
    
# ==========================================
# 3. 自定義層 (與論文一致)
# ==========================================
@keras.utils.register_keras_serializable()
class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, sequence_length, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.pos_emb = keras.layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)

    def call(self, inputs):
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        return inputs + self.pos_emb(positions)

    def get_config(self):
        config = super().get_config()
        config.update({"sequence_length": self.sequence_length, "embed_dim": self.embed_dim})
        return config

# ==========================================
# 4. 訊號處理與預測 (論文邏輯)
# ==========================================
def get_bpm_robust(window_data, fs=20):
    sig = window_data[:, 9] # Acc Mag
    if np.std(sig) < 0.005: return 0.0
    sig = sig - np.mean(sig)
    freqs = np.fft.rfftfreq(2048, d=1/fs)
    fft_mag = np.abs(np.fft.rfft(sig, n=2048))
    idx = np.where((freqs >= 0.08) & (freqs <= 0.85))[0] # 論文區間
    return freqs[idx][np.argmax(fft_mag[idx])] * 60 if len(idx) > 0 else 0.0

def clean_signal_logic(data, fs=20):
    cleaned_data = np.zeros_like(data)
    b, a = signal.butter(2, [0.08/(0.5*fs), 0.85/(0.5*fs)], btype='band')
    for i in range(data.shape[1]):
        f = signal.filtfilt(b, a, data[:, i])
        std = pd.Series(f).rolling(int(fs*10), center=True).std().bfill().ffill().replace(0, 1e-6)
        f_agc = f / std.values
        cleaned_data[:, i] = signal.savgol_filter(f_agc, 15, 2)
    return cleaned_data

if __name__ == "__main__":
    MODEL_PATH = "models/respiration_CNNxTran_classifier.keras"
    DATA_PATH = "data/StrawCompare/StaticSit_imu_YAHBOOM_20260320_002717.csv"
    
    print("\nStep 1: Hard-Patching Library Classes...")
    
    custom_objs = {
        'PositionalEmbedding': PositionalEmbedding,
        'MyLayers>PositionalEmbedding': PositionalEmbedding,
        'Functional': keras.Model
    }

    try:
        model = load_keras3_model_compat(MODEL_PATH, custom_objects=custom_objs)
        print("  [SUCCESS] Model loaded by bypassing Keras 2 internal crashes!")
    except Exception as e:
        print(f"  [ERROR] Load failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit()

    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df.columns = [c.lower().strip() for c in df.columns]
        time_col = 'unixtimestamp' if 'unixtimestamp' in df.columns else 'timestamp'
        df['dt'] = pd.to_datetime(df[time_col], unit='ms')
        df_res = df.set_index('dt').resample('50ms').mean().interpolate().ffill().bfill()
        
        acc_mag = np.linalg.norm(df_res[['accx', 'accy', 'accz']].values, axis=1, keepdims=True)
        gyro_mag = np.linalg.norm(df_res[['gyrox', 'gyroy', 'gyroz']].values, axis=1, keepdims=True)
        raw_feats = df_res.reindex(columns=['accx', 'accy', 'accz', 'gyrox', 'gyroy', 'gyroz', 'roll', 'pitch', 'yaw']).values
        combined = np.hstack([raw_feats, acc_mag, gyro_mag])
        cleaned = clean_signal_logic(combined, fs=20)

        print(f"\n{'Time':<6} | {'BPM':<5} | {'Status'}")
        print("-" * 35)
        
        for i in range(0, len(cleaned) - 200, 40):
            win = cleaned[i : i + 200]
            bpm = get_bpm_robust(win, fs=20)
            norm = (win - np.mean(win, axis=0)) / (np.std(win, axis=0) + 1e-6)
            prob = model.predict(norm.reshape(1, 200, 11), verbose=0)[0][0]
            
            status = "NORMAL"
            if bpm > 22.0 or (0.5 < bpm < 11.0) or prob > 0.0892:
                status = "ABNORMAL"
            print(f"{i/20:>5.1f}s | {bpm:>5.1f} | {status}")

    print("\nTask complete.")