

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# Load datasets
train_data = pd.read_csv('C:/Users/user/Desktop/a/balanced_train_data.csv')
test_data = pd.read_csv('C:/Users/user/Desktop/a/test_data.csv')

# Define label mapping
# label_mapping = {
#     0: 0, 1: 0,
#     2: 1, 3: 1,
#     4: 2, 5: 2,
#     6: 3
# }

# # Apply label mapping
# train_data['Launch price category'] = train_data['Launch price category'].map(label_mapping)
# test_data['Launch price category'] = test_data['Launch price category'].map(label_mapping)

# Separate features and target
X_train = train_data.drop(columns=['Launch price category'])
y_train = train_data['Launch price category']

X_test = test_data.drop(columns=['Launch price category'])
y_test = test_data['Launch price category']

# 自動區分類別與數值資料
categorical_features = [
    col for col in X_train.columns if X_train[col].max() <= 50 and 0 in X_train[col].unique()
]
numerical_features = X_train.columns.difference(categorical_features)

# 分開類別與數值資料
X_train_cat = X_train[categorical_features]
X_train_num = X_train[numerical_features]
X_test_cat = X_test[categorical_features]
X_test_num = X_test[numerical_features]

# 處理類別資料 (OneHotEncoding)
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_train_cat_encoded = encoder.fit_transform(X_train_cat)
X_test_cat_encoded = encoder.transform(X_test_cat)

# 處理數值資料 (MinMaxScaler)
scaler = MinMaxScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)

# 合併處理後的數據
X_train_processed = np.hstack((X_train_num_scaled, X_train_cat_encoded))
X_test_processed = np.hstack((X_test_num_scaled, X_test_cat_encoded))

# 定義超參數搜索範圍
param_grid = {
    'n_estimators': [110,120,130,140,150,200,1000,2000],
    'max_depth': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'min_samples_split': [2,3,4,5],
    'min_samples_leaf': [1]
}

# 使用 GridSearchCV 搜索最佳參數
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

# 對訓練資料進行超參數調整
grid_search.fit(X_train_processed, y_train)

# 獲取最佳參數
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# 使用最佳參數重新訓練模型
best_rf_model = grid_search.best_estimator_
best_rf_model.fit(X_train_processed, y_train)

# 測試模型並生成分類報告
y_pred = best_rf_model.predict(X_test_processed)
report = classification_report(y_test, y_pred, target_names=[str(c) for c in sorted(y_train.unique())])
print(report)

# 1. 計算 AUC ROC
y_pred_proba = best_rf_model.predict_proba(X_test_processed)
auc_roc = roc_auc_score(y_test, y_pred_proba, multi_class='ovo')
print(f"AUC ROC: {auc_roc}")

# 2. 提取並顯示前 10 特徵重要性
importances = best_rf_model.feature_importances_
feature_names = numerical_features.tolist() + encoder.get_feature_names_out(categorical_features).tolist()
important_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]

print("Top 10 Features:")
for name, score in important_features:
    print(f"{name}: {score:.4f}")

# 3. 視覺化特徵重要性
plt.figure(figsize=(10, 6))
plt.bar([name for name, _ in important_features], [score for _, score in important_features])
plt.xticks(rotation=45, ha='right')
plt.title('Top 10 Feature Importances')
plt.show()

# 視覺化混淆矩陣
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y_train.unique()))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# 視覺化 ROC 曲線
plt.figure(figsize=(10, 6))
for i in range(len(best_rf_model.classes_)):
    fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_pred_proba[:, i])
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc_score((y_test == i).astype(int), y_pred_proba[:, i]):.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
