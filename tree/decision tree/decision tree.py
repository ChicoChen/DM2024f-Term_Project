# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler
# # 讀取資料
# train_data = pd.read_csv('C:/Users/user/Desktop/a/balanced_train_data.csv')
# test_data = pd.read_csv('C:/Users/user/Desktop/a/test_data.csv')

# # Define label mapping
# label_mapping = {
#     0: 0, 1: 0,
#     2: 1, 3: 1,
#     4: 2, 5: 2,
#     6: 3
# }

# # Apply label mapping
# train_data['Launch price category'] = train_data['Launch price category'].map(label_mapping)
# test_data['Launch price category'] = test_data['Launch price category'].map(label_mapping)

# # 分離特徵與目標變數
# X_train = train_data.drop(columns=['Launch price category'])
# y_train = train_data['Launch price category']

# X_test = test_data.drop(columns=['Launch price category'])
# y_test = test_data['Launch price category']

# # 找出類別特徵 (值包含 0 的欄位) 與數值特徵
# categorical_features = [col for col in X_train.columns if (X_train[col] == 0).any()]
# numerical_features = [col for col in X_train.columns if col not in categorical_features]

# # 對數值特徵進行 Min-Max 正規化
# scaler = StandardScaler()
# X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
# X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# # 對類別特徵進行 One-Hot Encoding
# X_train_encoded = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)
# X_test_encoded = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)

# # 確保訓練集與測試集的特徵一致
# X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

# # 初始化並訓練 Decision Tree 模型
# dt_model = DecisionTreeClassifier(
#     max_depth=14,  # 限制深度
#     min_samples_split=4,  # 每個節點至少包含 n 個樣本才能進行分裂
#     min_samples_leaf=1,  # 每個葉子節點至少保留 n 個樣本
#     random_state=42
# )
# dt_model.fit(X_train_encoded, y_train)

# # 預測
# y_pred = dt_model.predict(X_test_encoded)

# # 評估模型
# accuracy = accuracy_score(y_test, y_pred)
# target_names = [str(class_name) for class_name in y_train.unique()]
# classification_rep = classification_report(y_test, y_pred, target_names=target_names)


# print("Numerical Features:", numerical_features)
# print("\nAccuracy:", accuracy)
# print("\nClassification Report:\n", classification_rep)
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# 讀取資料
train_data = pd.read_csv('C:/Users/user/Desktop/a/balanced_train_data.csv')
test_data = pd.read_csv('C:/Users/user/Desktop/a/test_data.csv')

# # Define label mapping
# label_mapping = {
#     0: 0, 1: 0,
#     2: 1, 3: 1,
#     4: 2, 5: 2,
#     6: 3
# }

# # Apply label mapping
# train_data['Launch price category'] = train_data['Launch price category'].map(label_mapping)
# test_data['Launch price category'] = test_data['Launch price category'].map(label_mapping)

# 分離特徵與目標變數
X_train = train_data.drop(columns=['Launch price category'])
y_train = train_data['Launch price category']

X_test = test_data.drop(columns=['Launch price category'])
y_test = test_data['Launch price category']

# 找出類別特徵 (值包含 0 的欄位) 與數值特徵
categorical_features = [col for col in X_train.columns if (X_train[col] == 0).any()]
numerical_features = [col for col in X_train.columns if col not in categorical_features]

# 對數值特徵進行 Min-Max 正規化
scaler = MinMaxScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# 對類別特徵進行 One-Hot Encoding
X_train_encoded = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)

# 確保訓練集與測試集的特徵一致
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

# 初始化並訓練 Decision Tree 模型
dt_model = DecisionTreeClassifier(
    max_depth=14,  # 限制深度
    min_samples_split=4,  # 每個節點至少包含 n 個樣本才能進行分裂
    min_samples_leaf=1,  # 每個葉子節點至少保留 n 個樣本
    random_state=42
)
dt_model.fit(X_train_encoded, y_train)

# 預測
y_pred = dt_model.predict(X_test_encoded)
y_pred_proba = dt_model.predict_proba(X_test_encoded)  # 預測機率

# 計算 AUC-ROC 分數與曲線
auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')  # 支援多分類
print("\nAUC-ROC Score:", auc_score)


for i in range(len(dt_model.classes_)):  # 使用模型中的類別數
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, i], pos_label=i)
    auc = roc_auc_score((y_test == i).astype(int), y_pred_proba[:, i])
    plt.plot(fpr, tpr, lw=2, label=f"Class {i} AUC = {auc:.2f}")

# 添加標籤與圖例
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for All Class")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# 評估模型
accuracy = accuracy_score(y_test, y_pred)
target_names = [str(class_name) for class_name in y_train.unique()]
classification_rep = classification_report(y_test, y_pred, target_names=target_names)

print("\nAccuracy:", accuracy)
print("\nClassification Report:\n", classification_rep)

# 提取 Top 10 Features
feature_importances = pd.DataFrame({
    'Feature': X_train_encoded.columns,
    'Importance': dt_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

top_10_features = feature_importances.head(10)
print("\nTop 10 Features:\n", top_10_features)

# 繪製 Top 10 Features 的圖表
plt.figure(figsize=(10, 6))
plt.barh(top_10_features['Feature'], top_10_features['Importance'], color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Top 10 Important Features")
plt.show()

