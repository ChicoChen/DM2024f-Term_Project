from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
import argparse
import sys
import pandas as pd
def bins(categories):
    categories=categories.replace({
        0: 0, 1: 0,
        2: 1, 3: 1,
        4: 2, 5: 2,
        6: 3
    })
    return categories
def loaddata(path='bpruned.csv', bin4=False):   
    train_data=pd.read_csv(path)
    test_data=pd.read_csv('bprtest.csv')

    X_train = train_data.drop(columns=['Launch price category'])
    y_train = train_data['Launch price category']
    if bin4:
        y_train = bins(y_train)
    X_test = test_data.drop(columns=['Launch price category'])
    y_test = test_data['Launch price category']
    if bin4:
        y_test = bins(y_test)

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    return X_train, y_train, X_test, y_test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", dest="input", help="filename containing data", default='bpruned.csv')
    parser.add_argument("-test", dest="testinput", help="filename containing data", default='bprtest.csv')
    parser.add_argument("--FourBins", action="store_true", default=False, help="Use 4 bins instead of 7")
    
    args = parser.parse_args()
    path = None
    if args.input is None:
        path = sys.stdin
    elif args.input is not None:
        path = args.input
    else:
        print("No dataset filename specified, system with exit\n")
        sys.exit("System will exit")
    if args.FourBins:
        bin4=True
    else:
        bin4=False
    X_train, y_train, X_test, y_test=loaddata(path, bin4)
    knn = KNeighborsClassifier(n_neighbors=5)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []

    for train_idx, val_idx in kfold.split(X_train, y_train):

        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
        knn.fit(X_fold_train, y_fold_train)
    
        y_fold_pred = knn.predict(X_fold_val)

        fold_accuracy = accuracy_score(y_fold_val, y_fold_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_fold_val, y_fold_pred, average='macro')
    
        fold_metrics.append({
            "Accuracy": fold_accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
        })

    cv_metrics_df = pd.DataFrame(fold_metrics)
    mean_cv_metrics = cv_metrics_df.mean()

    print("Cross-Validation Metrics (Mean):")
    print(mean_cv_metrics)

    knn.fit(X_train, y_train)

    y_test_pred = knn.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average='macro'
    )

    lb = LabelBinarizer()
    y_test_binary = lb.fit_transform(y_test)
    y_test_pred_binary = lb.transform(y_test_pred)

    test_roc_auc_scores = {}
    for i, class_label in enumerate(lb.classes_):
        test_roc_auc_scores[class_label] = roc_auc_score(
            y_test_binary[:, i], y_test_pred_binary[:, i]
        )
    test_metrics_summary = {
        "ROC AUC Scores": test_roc_auc_scores,
    }
    test_metrics_summary=pd.DataFrame(test_metrics_summary)
    print("\nTest Set Metrics:")
    print(f'Accuracy: {test_accuracy} \n Precision (macro avg): {test_precision} \n Recall (macro avg): {test_recall} \n F1-Score (macro avg): {test_f1}')
    print(test_metrics_summary)

if __name__ == "__main__":
    main()