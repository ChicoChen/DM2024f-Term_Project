import pandas as pd
import time
import pickle
import argparse
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.model_selection import RepeatedKFold
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, classification_report

import balance

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save an SVM model.")
    parser.add_argument("modelName", type=str, help="The filename to save the trained model")
    args = parser.parse_args()

    trainData_path = "./balanced_train_data_4Bins.csv"
    testData_path = "./test_data_4Bins.csv"

    trainingData = pd.read_csv(trainData_path)
    X = trainingData.drop(columns=["Launch price category"])
    Y = trainingData["Launch price category"]
    print("data loaded successfully")
    
    minmax = MinMaxScaler()
    # X = minmax.fit_transform(X)
    X = pd.DataFrame(minmax.fit_transform(X), columns=X.columns)

    RKF = RepeatedKFold(n_splits=5, n_repeats=1)
    svm = LinearSVC()
    # svm = SVC(kernel="linear", probability=True)

    accuracies = []
    reports = []
    count = 0
    cv_start_time = time.time()
    #cross validation
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
        reports.append(pd.DataFrame(report).transpose())
    
    # print CV average score
    average_accuracy = sum(accuracies) / len(accuracies)
    print("Average accuracy:", average_accuracy)
    df_reports = pd.concat(reports).groupby(level=0).mean()
    print("Average classification metrics across folds:\n", df_reports)

    cv_end_time = time.time()
    cv_training_time = cv_end_time - cv_start_time
    print(f"Cross-validation complete, total {cv_training_time:.2f}.\n")

    #train model on whole data
    svm.fit(X, Y)
    FT_time = time.time() - cv_end_time
    print(f"Full training complete, total {FT_time:.2f}.\nNow evaluating on the testing dataset...")
    
    #model testing
    testingData = pd.read_csv(testData_path)
    X_test = testingData.drop(columns=['Launch price category'])
    # X_test = minmax.fit_transform(X_test)
    X_test = pd.DataFrame(minmax.fit_transform(X_test), columns=X_test.columns)


    Y_test = testingData['Launch price category']
    Y_pred = svm.predict(X_test)
    test_accuracy = accuracy_score(Y_test, Y_pred)
    test_report = classification_report(Y_test, Y_pred)
    print("Testing dataset accuracy: ", test_accuracy)
    print("Testing dataset classification report:\n", test_report)
    
    with open("./models/" + args.modelName + ".pkl", 'wb') as model_file:
        pickle.dump(svm, model_file)
    print(f"Model saved as {args.modelName}.pkl")

