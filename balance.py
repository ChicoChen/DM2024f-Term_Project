import pandas as pd
from imblearn.over_sampling import SMOTE

def process_and_balance_data(file_path):

    # Load the data
    data = pd.read_csv(file_path)

    # Drop rows where the target variable 'Launch price category' is missing
    data = data.dropna(subset=['Launch price category'])

    # Separate features and target variable
    X = data.drop(columns=['Launch price category'])
    y = data['Launch price category']

    # Drop the first column in X
    X = X.iloc[:, 1:]

    # Apply SMOTE to balance the 'Launch price category' classes
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Combine resampled features and target variable into a single DataFrame
    balanced_data = pd.concat([X_resampled, y_resampled.rename('Launch price category')], axis=1)

    return balanced_data

# Example usage
# file_path = 'C:/Users/user/Downloads/processed(merged version).csv'
# balanced_data = process_and_balance_data(file_path)
# print(balanced_data.head())  # Display the first few rows of the balanced data
