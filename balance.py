import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RepeatedKFold

def process_and_balance_data(file_path):
    # Load the data
    data = pd.read_csv(file_path)

    # Drop rows where the target variable 'Launch price category' is missing
    data = data.dropna(subset=['Launch price category'])

    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Separate features and target variable
    X = data.drop(columns=['Launch price category'])
    y = data['Launch price category']

    # Drop the first column in X
    X = X.iloc[:, 1:]

    # Initialize RepeatedKFold with an 8:2 split
    rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)

    # Iterate over splits
    for train_index, test_index in rkf.split(X):
        # Split data into training and testing sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Apply SMOTE to balance the training data
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        # Combine resampled features and target variable into a DataFrame for the balanced training data
        balanced_train_data = pd.concat([X_train_balanced, y_train_balanced.rename('Launch price category')], axis=1)
        
        # Create a DataFrame for the test data
        test_data = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
        
        # Return balanced training data and test data
        return balanced_train_data, test_data
    
# Example usage
def main():
    # File path for the input CSV
    file_path = './data/prunedB.csv'
    
    # Process and balance data
    balanced_train_data, test_data = process_and_balance_data(file_path)
    
    # Save the balanced training data and test data to CSV files
    balanced_train_data.to_csv('./balanced_train_data.csv', index=False)
    test_data.to_csv('./test_data.csv', index=False)


if __name__ == "__main__":
    main()
