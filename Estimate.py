import numpy as np
from sklearn.linear_model import LinearRegression

def predict_next_month_inventory(stock_levels, sales):
    # 構建特徵矩陣 (X) 和目標變數 (y)
    X = np.array(stock_levels).reshape(-1, 1)  # 轉換成列向量
    y = np.array(sales)  # 銷售量作為目標變數
    
    # 建立線性回歸模型
    model = LinearRegression()
    model.fit(X, y)
    
    # 預測下一個月的銷售量
    next_month_stock = stock_levels[-1] - sales[-1]  # 預測下個月起始庫存
    predicted_sales = model.predict(np.array([[next_month_stock]]))[0]
    
    # 計算需要的進貨量，確保庫存不低於前一個月的水準
    required_stock = predicted_sales - next_month_stock
    required_stock = max(0, required_stock)  # 進貨量不可為負
    
    return round(required_stock, 2)

# 測試範例
stock_levels = [50, 45, 40]  # 前三個月庫存量 (萬單位)
sales = [10, 12, 15]  # 前三個月銷售量 (萬單位)

next_month_purchase = predict_next_month_inventory(stock_levels, sales)
print(f"下個月需要進貨量: {next_month_purchase} 萬單位")
