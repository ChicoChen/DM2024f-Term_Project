import pandas as pd
import time
import pickle
import argparse
from sklearn.model_selection import RepeatedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

import balance

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save an SVM model.")
    parser.add_argument("modelName", type=str, help="The filename to save the trained model")
    args = parser.parse_args()

    trainData_path = "./balanced_train_data.csv"
    testData_path = "./test_data.csv"

    trainingData = pd.read_csv(trainData_path)
    X = trainingData.drop(columns=["Launch price category"])
    Y = trainingData["Launch price category"]
    print("data loaded successfully")

    RKF = RepeatedKFold(n_splits=5, n_repeats=1)
    svm = SVC(kernel="linear", probability=True)

    accuracies = []
    reports = []
    count = 0
    cv_start_time = time.time()
    for trainIdx, valIdx in RKF.split(trainingData):
        X_train, X_val = X.iloc[trainIdx], X.iloc[valIdx]
        y_train, y_val = Y.iloc[trainIdx], Y.iloc[valIdx]
        
        print(f"begin training, iteration: {count}")
        count += 1
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_val)
        
        print("iteration done, appending accuracy")
        accuracy = accuracy_score(y_val, y_pred)
        accuracies.append(accuracy)
        
        report = classification_report(y_val, y_pred, output_dict=True)
        reports.append(report)

    average_accuracy = sum(accuracies) / len(accuracies)
    print("Average accuracy:", average_accuracy)
    df_reports = pd.DataFrame(reports).mean()
    print("Average classification metrics across folds:\n", df_reports)

    cv_end_time = time.time()
    cv_training_time = cv_end_time - cv_start_time
    print(f"Cross-validation complete, total{cv_training_time:.2f}.\nNow evaluating on the testing dataset...")

    svm.fit(X, Y)
    FT_time = time.time() - cv_end_time
    print(f"Full training complete, total{FT_time:.2f}.\nNow evaluating on the testing dataset...")
    
    testingData = pd.read_csv(trainData_path)
    X_test = testingData.drop(columns=['Launch price category'])
    Y_test = testingData['Launch price category']
    Y_pred = svm.predict(X_test)
    test_accuracy = accuracy_score(Y_test, Y_pred)
    test_report = classification_report(Y_test, Y_pred)
    print("Testing dataset accuracy: ", test_accuracy)
    print("Testing dataset classification report:\n", test_report)
    
    with open(args.modelName + ".pkl", 'wb') as model_file:
        pickle.dump(svm, model_file)
    print(f"Model saved as {args.modelName}.pkl")


