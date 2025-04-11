import bluetooth
import time

def find_bluetooth_device(name):
    """尋找配對的藍牙裝置名稱"""
    devices = bluetooth.discover_devices(lookup_names=True)
    for addr, device_name in devices:
        if device_name == name:
            return addr
    return None

def consume_bluetooth_keyboard_power(device_name, interval=0.5):
    """持續發送 HID 信號模擬按鍵輸入"""
    print(f"搜尋裝置 {device_name}...")
    device_address = find_bluetooth_device(device_name)

    if not device_address:
        print(f"找不到裝置 {device_name}。請確保藍牙配對已完成並開啟電源。")
        return

    print(f"已找到裝置：{device_name} ({device_address})")
    
    # 嘗試建立藍牙 RFCOMM 連接
    try:
        sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        sock.connect((device_address, 1))  # 通常 HID 使用 RFCOMM 通道 1
        print("已連接到藍牙裝置。開始模擬鍵盤輸入...")

        while True:
            # HID 信號 (這裡模擬按下並鬆開 "A" 鍵)
            sock.send(b'\x00\x00\x04\x00\x00\x00\x00\x00')  # 按下 "A"
            time.sleep(interval)
            sock.send(b'\x00\x00\x00\x00\x00\x00\x00\x00')  # 鬆開 "A"
            time.sleep(interval)

    except bluetooth.BluetoothError as e:
        print(f"藍牙連接失敗：{e}")
    finally:
        sock.close()
        print("連接已關閉。")

if __name__ == "__main__":
    keyboard_name = input("輸入藍牙鍵盤名稱: ")
    consume_bluetooth_keyboard_power(keyboard_name)