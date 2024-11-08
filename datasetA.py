import pandas as pd
import re
# 讀取 CSV 檔案
file_path = 'C:/Users/user/Downloads/nanoreview_phone_specs (processed).csv'
df = pd.read_csv(file_path)

df['Brand'] = pd.Categorical(df['Brand'], categories=df['Brand'].unique(), ordered=True)
df['Brand'] = df['Brand'].cat.codes
df['Type'] = pd.Categorical(df['Type'], categories=df['Type'].unique(), ordered=True)
df['Type'] = df['Type'].cat.codes
df['Max rated brightness'].fillna(500, inplace=True)
df['Touch sampling rate'].fillna(120, inplace=True)
# 將 'Display features' 欄位中的空值填為空字串，以便後續處理
df['Display features'] = df['Display features'].fillna('')
def convert_waterproof(value):
    if isinstance(value, str):  # 確保只有在值是字串時才執行正則表達式
        if value == "Yes":
            return 1
        elif value == "No":
            return 0
        else:
            # 使用正則表達式提取 IP 等級中的數字部分（區分大小寫）
            match = re.search(r'IP(\d+)', value)
            if match:
                return int(match.group(1)[-1])  # 提取最後一個數字作為防水等級
    # 如果值不是字串或不符合上述條件，則返回 0
    return 0

# 將 'Waterproof' 欄位轉換
df['Waterproof'] = df['Waterproof'].apply(convert_waterproof)
df['Rear material'] = df['Rear material'].map({
    'Plastic': 1,
    'Glass': 2,
    'Metal': 3,
    'Eco-leather': 4
}).fillna(0).astype(int)  # 將 NaN 填入 0，並轉換為整數型態

df['Frame material'] = df['Frame material'].map({
    'Plastic': 1,
    'Metal': 2
}).fillna(0).astype(int) 

df['Fingerprint scanner'] = df['Fingerprint scanner'].map({
    'Yes, in-display': 1,
    'Yes, in-button': 2,
    'Yes, rear': 3,
    'No': 0
}).fillna(0).astype(int)  # 將 NaN 填入 0，並轉換為整數型態
df['L3 cache (mb)'].fillna(0, inplace=True)
df['Manufacturing'] = df['Manufacturing'].map({
    'TSMC': 1,
    'Samsung': 2,
    'SMIC': 3,
}).fillna(0).astype(int)
# 使用 str.get_dummies() 來將多重選項拆分成多個欄位
display_features_dummies = df['Display features'].str.get_dummies(sep=' - ')

# 將新產生的 dummy 欄位合併回原始資料框
df = pd.concat([df, display_features_dummies], axis=1)
df.drop(columns=['Display features'], inplace=True)
# 儲存結果到新的 CSV 檔案
output_file_path = 'C:/Users/user/Downloads/updated_nanoreview_phone_specs.csv'
df.to_csv(output_file_path, index=False)

# 顯示結果

print(f"已將含有空值的欄位儲存為 {output_file_path}")
