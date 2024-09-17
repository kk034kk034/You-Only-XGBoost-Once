import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def load_and_clean_data(file_path):
    """
    加載並清理Excel數據
    """
    try:
        # 讀取Excel文件，跳過前7行元數據
        data = pd.read_excel(file_path, sheet_name='工作表2', skiprows=7, header=None)
        
        # 根據數據結構手動指定列名
        data.columns = ['order_id', 'member_id', 'postal_cd', 'date_cd', 'time_cd', 'department', 'goods', 'prod_id', 'price', 'prime']
        
        # 移除所有全為NaN的列
        data.dropna(axis=1, how='all', inplace=True)
        
        # 移除所有全為NaN的行
        data.dropna(axis=0, how='all', inplace=True)
        
        # 填補缺失值
        data.fillna({
            'member_id': 'Unknown',
            'postal_cd': '000',
            'date_cd': '1970-01-01',
            'time_cd': '00:00:00',
            'department': 'Unknown',
            'goods': 'Unknown',
            'prod_id': 'Unknown',
            'price': 0,
            'prime': 'No'
        }, inplace=True)
        
        # 將日期和時間轉換為datetime格式
        data['date_cd'] = pd.to_datetime(data['date_cd'], errors='coerce')
        data['time_cd'] = pd.to_datetime(data['time_cd'], format='%H:%M:%S', errors='coerce').dt.time
        
        # 處理Prime會員欄位，將其轉換為二進制
        data['prime'] = data['prime'].apply(lambda x: 1 if x.strip().lower() == 'yes' else 0)
        
        return data
    except Exception as e:
        print(f"數據加載或清理過程中出現錯誤: {e}")
        return None

def analyze_shopping_cycle(data):
    """
    分析消費者的購物周期
    """
    # 計算每個會員的購買次數和購買間隔
    data_sorted = data.sort_values(by=['member_id', 'date_cd'])
    data_sorted['previous_date'] = data_sorted.groupby('member_id')['date_cd'].shift(1)
    data_sorted['days_between'] = (data_sorted['date_cd'] - data_sorted['previous_date']).dt.days
    
    # 計算平均購買間隔
    avg_cycle = data_sorted.groupby('member_id')['days_between'].mean().reset_index()
    avg_cycle.rename(columns={'days_between': 'avg_purchase_cycle_days'}, inplace=True)
    
    return avg_cycle

def predict_preferred_products(data):
    """
    預測消費者偏好的商品類別
    """
    # 特徵工程：提取日期相關特徵
    data['month'] = data['date_cd'].dt.month
    data['day_of_week'] = data['date_cd'].dt.dayofweek
    
    # 目標變量：商品類別
    X = data[['price', 'prime', 'month', 'day_of_week']]
    y = data['department']
    
    # 編碼目標變量
    y_encoded = y.factorize()[0]
    
    # 分割數據集
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # 訓練隨機森林分類器
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # 預測
    y_pred = clf.predict(X_test)
    
    # 評估模型
    print("分類報告:")
    print(classification_report(y_test, y_pred))
    print("混淆矩陣:")
    print(confusion_matrix(y_test, y_pred))
    
    return clf

def main():
    # 定義Excel文件的路徑
    file_path = 'PChome_日用品銷售_樣本.xlsx'
    
    # 加載並清理數據
    data = load_and_clean_data(file_path)
    if data is None:
        return
    
    print("數據清理完成，數據預覽:")
    print(data.head())
    
    # 分析購物周期
    avg_cycle = analyze_shopping_cycle(data)
    print("每位會員的平均購買周期（天）:")
    print(avg_cycle.head())
    
    # 預測消費者偏好的商品類別
    clf = predict_preferred_products(data)
    
    # # 列出所有可用字體
    # for font in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
    #     print(fm.FontProperties(fname=font).get_name())

    # 使用 SimHei（黑體）字體顯示中文
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者使用 ['Malgun Gothic']
    plt.rcParams['axes.unicode_minus'] = False    # 正常顯示負號
    # 可視化一些分析結果
    plt.figure(figsize=(10,6))
    sns.histplot(avg_cycle['avg_purchase_cycle_days'], bins=30, kde=True)
    plt.title('會員平均購買周期分佈')
    plt.xlabel('平均購買周期（天）')
    plt.ylabel('會員數量')
    plt.show()

if __name__ == "__main__":
    main()
