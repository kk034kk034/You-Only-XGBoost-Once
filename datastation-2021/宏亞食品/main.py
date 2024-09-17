import pandas as pd
import json
import re
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from fuzzywuzzy import process
import warnings

# 忽略不必要的警告
warnings.filterwarnings('ignore')

def clean_name(name):
    """
    清理商品名稱或品名，移除非中文字符並轉換為小寫。
    """
    if pd.isnull(name):
        return ''
    # 移除所有非中文字符
    cleaned = re.sub(r'[^\u4e00-\u9fff]', '', name)
    return cleaned.lower()

def load_sales_data(csv_file):
    """
    加載銷售數據的CSV文件
    :param csv_file: CSV文件的路徑
    :return: 解析後的銷售數據
    """
    sales_data = pd.read_csv(csv_file, encoding='utf-8')
    return sales_data

def load_food_data(json_file):
    """
    加載食品成分標示的JSON文件
    :param json_file: JSON文件的路徑
    :return: 解析後的食品數據
    """
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return pd.DataFrame(data)

def preprocess_data(sales_data, food_data):
    """
    結合銷售數據和食品成分數據，進行預處理
    :param sales_data: 銷售數據
    :param food_data: 食品數據
    :return: 結合後的數據
    """
    # 清理 '商品名稱' 和 '品名' 列，去除多餘空格並統一為小寫，移除非中文字符
    sales_data['商品名稱_cleaned'] = sales_data['商品名稱'].str.strip().apply(clean_name)
    food_data['品名_cleaned'] = food_data['品名'].str.strip().apply(clean_name)

    # 打印一些清理後的值進行檢查
    print("清理後的示例商品名稱（前10條）:", sales_data['商品名稱_cleaned'].head(10).tolist())
    print("清理後的示例品名（前10條）:", food_data['品名_cleaned'].head(10).tolist())

    # 找到共同的商品名稱
    common_names = set(sales_data['商品名稱_cleaned']).intersection(set(food_data['品名_cleaned']))
    print(f"共有 {len(common_names)} 個共同的商品名稱和品名。")

    if len(common_names) > 0:
        print("一些共同的商品名稱和品名示例：", list(common_names)[:10])
        # 直接合併
        combined_data = pd.merge(sales_data, food_data, how='inner',
                                 left_on='商品名稱_cleaned', right_on='品名_cleaned')
    else:
        print("沒有找到共同的商品名稱和品名。開始進行模糊匹配。")
        # 使用模糊匹配
        food_names = food_data['品名_cleaned'].tolist()

        def match_name(name, food_names, threshold=80):
            if not name:
                return None
            match = process.extractOne(name, food_names)
            if match and match[1] >= threshold:
                return match[0]
            else:
                return None

        sales_data['mapped品名'] = sales_data['商品名稱_cleaned'].apply(
            lambda x: match_name(x, food_names) if x else None)

        # 移除無法匹配的行
        sales_data_matched = sales_data[sales_data['mapped品名'].notnull()]
        print(f"模糊匹配後有 {len(sales_data_matched)} 條記錄。")

        # 合併數據
        combined_data = pd.merge(sales_data_matched, food_data, how='inner',
                                 left_on='mapped品名', right_on='品名_cleaned')

    if combined_data.empty:
        print("合併後的數據仍然為空。請檢查數據來源和合併鍵。")
        return combined_data

    # 提取食品成分作為特徵，並轉換為數值特徵
    combined_data['is_chocolate'] = combined_data['品名_cleaned'].apply(lambda x: 1 if '巧克力' in x else 0)
    combined_data['is_cookie'] = combined_data['品名_cleaned'].apply(lambda x: 1 if '餅乾' in x else 0)
    combined_data['vegetarian'] = combined_data['葷素類別'].apply(lambda x: 1 if x == '素' else 0)

    # 填補缺失值
    combined_data.fillna(0, inplace=True)

    # 打印檢查合併後的數據
    print(f"合併後的數據有 {len(combined_data)} 條記錄")
    print(combined_data.head())

    return combined_data

def train_xgboost_model(data):
    """
    訓練XGBoost模型來預測產品銷量
    :param data: 預處理過的數據
    :return: 訓練好的模型及預測結果
    """
    # 選擇特徵和目標變量
    X = data[['is_chocolate', 'is_cookie', 'vegetarian', '單價']]  # 可以添加更多特徵
    y = data['數量']  # 假設 '數量' 是我們的目標變量

    print(f"X（特徵）的形狀: {X.shape}")
    print(f"y（目標）的形狀: {y.shape}")

    # 分割訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    if len(X_train) == 0:
        print("訓練集為空，請檢查數據是否正確！")
        return None, None, None

    # 轉換為DMatrix格式
    train_dmatrix = xgb.DMatrix(X_train, label=y_train)
    test_dmatrix = xgb.DMatrix(X_test, label=y_test)

    # 設置XGBoost參數
    params = {
        'objective': 'reg:squarederror',  # 回歸問題
        'max_depth': 6,
        'eta': 0.3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'rmse'
    }

    # 訓練模型
    model = xgb.train(params, train_dmatrix, num_boost_round=100,
                      evals=[(test_dmatrix, 'eval')],
                      early_stopping_rounds=10)

    # 預測測試集
    y_pred = model.predict(test_dmatrix)

    # 評估模型
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"測試集的 RMSE: {rmse}")

    return model, y_test, y_pred

def main():
    # CSV和JSON文件的路徑
    csv_file_path = 'Hunya_pchome.csv'
    json_file_path = 'Hunya_食品成分標示.json'

    # 加載銷售數據和食品數據
    sales_data = load_sales_data(csv_file_path)
    food_data = load_food_data(json_file_path)
    print("銷售數據的列名：", sales_data.columns)
    print("食品數據的列名：", food_data.columns)

    # 數據預處理
    combined_data = preprocess_data(sales_data, food_data)

    if combined_data is not None and not combined_data.empty:
        # 訓練模型並進行預測
        model, y_test, y_pred = train_xgboost_model(combined_data)

        # 如果模型成功訓練，打印模型的預測結果
        if model is not None:
            print("預測完成！測試集的真實數量和預測數量：")
            print(pd.DataFrame({'真實數量': y_test, '預測數量': y_pred}))
    else:
        print("合併數據後為空，請檢查數據是否匹配正確！")

if __name__ == "__main__":
    main()
