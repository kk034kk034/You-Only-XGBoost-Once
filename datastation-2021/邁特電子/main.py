import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

# 忽略不必要的警告
warnings.filterwarnings('ignore')

# 讀取數據
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    print("數據集中的欄位名稱：", data.columns)
    return data

# 計算安全庫存量
def calculate_safe_stock(order_qty, safety_factor=1.5):
    """
    計算安全庫存量的示例函數。
    您可以根據業務需求調整此函數。
    """
    return order_qty * safety_factor

# 數據預處理
def preprocess_data(data):
    # 填補缺失值
    data.fillna(0, inplace=True)
    
    # 將重要性列轉換為數值類別（假設重要性為'A', 'B', 'C'}
    # 根據常見的 ABC 分類，通常 A=3（高），B=2（中），C=1（低）
    importance_mapping = {'A': 3, 'B': 2, 'C': 1}
    data['QtyABC_numeric'] = data['QtyABC'].map(importance_mapping)
    data['ValueABC_numeric'] = data['ValueABC'].map(importance_mapping)
    
    # 處理 VendorCode，如果需要，可以進行編碼
    data['VendorCode_encoded'] = data['VendorCode'].astype('category').cat.codes
    
    # 計算安全庫存量
    data['安全庫存量'] = data['AvgOfSumOfAdjQty1'].apply(lambda x: calculate_safe_stock(x))
    
    return data

# 訓練XGBoost模型
def train_xgboost_model(data):
    # 確認目標變量
    target_column = '安全庫存量'
    if target_column not in data.columns:
        print(f"目標變量 '{target_column}' 不存在於數據集中。請確認目標變量名稱或添加目標變量。")
        return None
    
    # 提取特徵和目標變量
    X = data[['AvgOfSumOfAdjQty1', 'QtyABC_numeric', 'ValueABC_numeric', 'VendorCode_encoded']]
    y = data[target_column]
    
    print("特徵和目標變量的統計信息：")
    print(X.describe())
    print(y.describe())
    
    # 分割訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # 構建XGBoost模型
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
    
    # 預測測試集
    y_pred = model.predict(X_test)
    
    # 計算均方誤差 (MSE) 和平均絕對誤差 (MAE)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"均方誤差 (MSE): {mse}")
    print(f"平均絕對誤差 (MAE): {mae}")
    
    return model, y_test, y_pred

# 主函數
def main():
    # 讀取數據
    csv_file = 'Mighty.csv'  # 請確保此文件存在並包含所需欄位
    data = load_data(csv_file)
    
    # 數據預處理
    processed_data = preprocess_data(data)
    
    # 訓練模型並進行預測
    model_result = train_xgboost_model(processed_data)
    
    if model_result:
        model, y_test, y_pred = model_result
        # 如果需要，可以進一步分析預測結果
        results = pd.DataFrame({'真實值': y_test, '預測值': y_pred})
        print("預測結果示例：")
        print(results.head())

if __name__ == "__main__":
    main()
