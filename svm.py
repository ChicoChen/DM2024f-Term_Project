import pandas as pd
import time
import pickle
import argparse
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save an SVM model.")
    parser.add_argument("modelName", type=str, help="The filename to save the trained model")
    parser.add_argument("--FourBins", action="store_true", default=False)
    args = parser.parse_args()
    print(args.FourBins)
    trainData_path = "./balanced_train_data.csv"
    testData_path = "./test_data.csv"

    trainingData = pd.read_csv(trainData_path)
    X = trainingData.drop(columns=["Launch price category"])
    Y = trainingData["Launch price category"]
    print("data loaded successfully")
    
    label_mapping = {
        0: 0, 1: 0,
        2: 1, 3: 1,
        4: 2, 5: 2,
        6: 3
    }

    if args.FourBins:
        Y = Y.replace(label_mapping)
    #data["Launch price category"] = data["Launch price category"].replace(label_mapping)

    unique_values = X.nunique()
    unique_values.to_csv("unique.csv")
    for col in X.columns:
        if unique_values[col] > 10:
            Q1 = X[col].quantile(0.25)  # First quartile
            Q3 = X[col].quantile(0.75)  # Third quartile
            IQR = Q3 - Q1  # Interquartile Range
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
            
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('svc', SVC(kernel="linear", probability=True, class_weight='balanced'))
        # ('xgb', XGBClassifier(eval_metric='logloss',
        #                     min_child_weight=3, gamma=0.1, eta=0.1))
        ])
    
    # XGB_param_grid = {
    #     'xgb__max_depth': [3, 5, 7],
    #     'xgb__learning_rate': [0.01, 0.1],
    #     'xgb__n_estimators': [50, 100, 200],
    #     'xgb__subsample': [0.6, 0.8, 1.0],
    #     'xgb__colsample_bytree': [0.6, 0.8, 1.0]
    # }
    param_grid = {
        'svc__C': [0.001, 0.01, 0.1, 1],
        'svc__tol': [0.001, 0.01, 0.1, 1],
        }

    grid_search = GridSearchCV(
                    pipeline,
                    param_grid,
                    cv=5,
                    scoring='accuracy',
                    n_jobs=-1
                )

    cv_start_time = time.time()
    grid_search.fit(X, Y)
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation accuracy:", grid_search.best_score_)
    
    cv_end_time = time.time()
    cv_training_time = cv_end_time - cv_start_time
    best_model = grid_search.best_estimator_
    print(f"Cross-validation complete, total {cv_training_time:.2f}s.\n")
    print(f"Now evaluating on the testing dataset...")
    
    #model testing
    testingData = pd.read_csv(testData_path)
    X_test = testingData.drop(columns=['Launch price category'])

    Y_test = testingData['Launch price category']
    if args.FourBins:
        Y_test = Y_test.replace(label_mapping)

    Y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.decision_function(X_test)
    Y_test_binarized = label_binarize(Y_test, classes=best_model.classes_)
    
    metrics = {
        'Accuracy': accuracy_score(Y_test, Y_pred),
        'Precision': precision_score(Y_test, Y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(Y_test, Y_pred, average='weighted', zero_division=0),
        'F1-score': f1_score(Y_test, Y_pred, average='weighted', zero_division=0),
        'AUC-ROC': roc_auc_score(Y_test_binarized, y_pred_proba, average='weighted', multi_class='ovr')
    }
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")

    test_report = classification_report(Y_test, Y_pred, zero_division=1)
    print("Testing dataset classification report:\n", test_report)
    
    # Calculate feature importance
    feature_names = X_test.columns.to_list()
    feature_importance = np.abs(best_model['svc'].coef_[0]) if len(best_model['svc'].coef_.shape) > 1 else np.abs(best_model['svc'].coef_)
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    print(f"top 10 features:\n{feature_importance_df.head(10)}")

    feature_importance_path = "./SVM/feature_importances"
    if args.FourBins:
        feature_importance_path = feature_importance_path + "_4Bins"
    feature_importance_df.to_csv(feature_importance_path + ".csv", index=False)

    modelPath = "./models/" + args.modelName
    if args.FourBins:
        modelPath = modelPath + "_4Bins"
    with open(modelPath + ".pkl", 'wb') as model_file:
        pickle.dump(modelPath + ".pkl", model_file)
    print(f"Model saved at {modelPath}.pkl")