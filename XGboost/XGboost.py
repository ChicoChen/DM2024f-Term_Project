import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import argparse
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from collections import Counter
import matplotlib.pyplot as plt


# preprocess
def preprocess_data(file_path, args):
    data = pd.read_csv(file_path)

    data = data.dropna(subset=['Launch price category'])

    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    label_mapping_4bins = {
        0: 0, 1: 0,
        2: 1, 3: 1,
        4: 2, 5: 2,
        6: 3
    }

    label_mapping_3bins = {
        0: 0, 1: 0, 2: 0,
        3: 1, 4: 1,
        5: 2, 6: 2
    }

    X = data.drop(columns=['Launch price category'])
    y = data['Launch price category']

    if args.FourBins:
        y = y.replace(label_mapping_4bins)
    elif args.ThreeBins:
        y = y.replace(label_mapping_3bins)

    return X, y


def main():
    parser = argparse.ArgumentParser(description="Train and optimize an XGBoost model using GridSearchCV.")
    parser.add_argument("--FourBins", action="store_true", default=False, help="Use 4 bins instead of 7")
    parser.add_argument("--ThreeBins", action="store_true", default=False, help="Use 3 bins instead of 7")
    args = parser.parse_args()

    file_path = "../data/pruned.csv"  
    X, y = preprocess_data(file_path, args)
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print("Before SMOTE:", Counter(y_train))
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print("After SMOTE:", Counter(y_train))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)


    model = xgb.XGBClassifier(objective="multi:softmax", eval_metric="mlogloss")

    param_grid = {
        "max_depth": [7],
        "learning_rate": [0.1],
        "n_estimators": [100],
        "subsample": [1.0],
        "colsample_bytree": [0.8]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        verbose=2,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Test set accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


    feature_importance = best_model.feature_importances_
    feature_names = X.columns

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })

    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    feature_importance_df.to_csv('feature_importance.csv', index=False)
    print("Feature importance saved to 'feature_importance.csv'")


    plt.figure(figsize=(10, 5))
    xgb.plot_tree(best_model, num_trees=0)
    plt.show()

    


if __name__ == "__main__":
    main()
