import pandas as pd
import time
import pickle
import argparse

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, classification_report

import balance

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
        ])
    param_grid = {
        'svc__C': [0.001, 0.01, 0.1, 1],
        'svc__tol': [0.001, 0.01, 0.1, 1]
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
    print(f"Cross-validation complete, total {cv_training_time:.2f}s.\n")
    
    best_model = grid_search.best_estimator_
    # best_model.fit(X, Y)
    FT_time = time.time() - cv_end_time
    print(f"Full training complete, total {FT_time:.2f}s.\nNow evaluating on the testing dataset...")
    
    #model testing
    testingData = pd.read_csv(testData_path)
    X_test = testingData.drop(columns=['Launch price category'])

    Y_test = testingData['Launch price category']
    if args.FourBins:
        Y_test = Y_test.replace(label_mapping)

    Y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(Y_test, Y_pred)
    test_report = classification_report(Y_test, Y_pred, zero_division=1)
    print("Testing dataset accuracy: ", test_accuracy)
    print("Testing dataset classification report:\n", test_report)
    
    modelPath = "./models/" + args.modelName
    if args.FourBins:
        modelPath = modelPath + "_4Bins"
    with open("./models/" + args.modelName + ".pkl", 'wb') as model_file:
        pickle.dump(modelPath + ".pkl", model_file)
    print(f"Model saved at {modelPath}.pkl")