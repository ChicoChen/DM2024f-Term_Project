import pandas as pd
from functools import reduce
import numpy as np
from sklearn.preprocessing import OneHotEncoder
#read csv and seperate it as a list of length 4;
PATH = "G:/nanoreview_phone_specs (processed).csv"
data = pd.read_csv("G:/nanoreview_phone_specs (processed).csv")
data_list= [data.iloc[:, i:i+27] for i in range(0, 108, 27)]

#TODO: process each portion of data
#************ Portion 1 ************

#************ Portion 2 ************

#************ Portion 3 ************
missing_values = data_list[2].isnull().sum()
unique_values = data_list[2].nunique()

# Apply the cut function to 'Launch price ($)' to categorize the prices
bins = [0, 250, 500, 750, 1000, 1250, 1500, float('inf')]
labels = [0, 1, 2, 3, 4, 5, 6]
data_list[2]['Launch price category'] = pd.cut(data_list[2]['Launch price ($)'], bins=bins, labels=labels, right=True, include_lowest=True)
data_list[2]=data_list[2].drop(columns=['Launch price ($)'])
# one value type & w/o missing value ->useless
data_list[2] = data_list[2].drop(columns=['Type of SIM card', 'Angle of widest lens (°)'])

# Extracting the p and FPS values
data_list[2][['Video resolution_p', 'Video resolution_FPS']] = data_list[2]['Video resolution'].str.extract(r'(\d+)p.*?(\d+) FPS')
data_list[2]=data_list[2].drop(columns=['Video resolution'])
new_columns = [col for col in data_list[2].columns if col.startswith('Video resolution_')]

# Yes/No columns-> 1/0
binary_columns = unique_values[unique_values == 2].index
for col in binary_columns:
    data_list[2][col] = data_list[2][col].replace({'Yes': 1, 'No': 0})

# the 'Multi SIM mode' column 'Active' -> 1 and 'Standby' -> 0
data_list[2]['Multi SIM mode'] = data_list[2]['Multi SIM mode'].replace({'Active': 1, 'Standby': 0})

columns_to_encode = ['8K video recording', '4K video recording', '1080p video recording']+new_columns
for column in columns_to_encode:
    data_list[2][column] = data_list[2][column].replace('No', 0)   
    unique_values = data_list[2][column][data_list[2][column] != 0].unique()
    # extract digits for sorting
    sorted_unique_values = sorted(unique_values, key=lambda x: int(''.join(filter(str.isdigit, str(x)))))
    encoding_map = {value: idx + 1 for idx, value in enumerate(sorted_unique_values)}
    
    data_list[2][column] = data_list[2][column].replace(encoding_map)

data_list[2]['NFC*'] = data_list[2]['NFC*'].replace({
        'No': 0,
        'Depends on the region': 1,
        'Yes': 2
    })

data_list[2]['Category'] = data_list[2]['Category'].replace({
        'Budget': 0,
        'Mid-range': 1,
        'Flagship': 2
    })
data_list[2]['Bokeh mode'] = data_list[2]['Camera features'].str.contains('Bokeh mode').astype(int)
data_list[2]['Pro mode'] = data_list[2]['Camera features'].str.contains('Pro mode').astype(int)
data_list[2]['RAW support'] = data_list[2]['Camera features'].str.contains('RAW support').astype(int)
data_list[2] = data_list[2].drop(columns=['Camera features'])

# OneHotEncoder
autofocus_column = data_list[2][['Autofocus', 'Speakers']]
encoder = OneHotEncoder(drop='if_binary', sparse=False)
autofocus_encoded = encoder.fit_transform(autofocus_column)
encoded_feature_names = encoder.get_feature_names_out(['Autofocus', 'Speakers'])
autofocus_encoded_df = pd.DataFrame(autofocus_encoded, columns=encoded_feature_names).drop(['Autofocus_No', 'Speakers_nan'], axis=1)
# Drop the original 'Autofocus' column and concatenate the encoded dataframe
data_list[2] = data_list[2].drop(['Autofocus','Speakers'], axis=1)
data_list[2] = pd.concat([data_list[2], autofocus_encoded_df], axis=1)

data_list[2]['Sensor size'].fillna(data_list[2]['Sensor size'].mean(), inplace=True)
data_list[2]['Pixel size (micron)'].fillna(data_list[2]['Pixel size (micron)'].mean(), inplace=True)

for column in data_list[2].columns:
    if column not in ['Sensor size', 'Pixel size (micron)','Launch price category']:
        data_list[2][column].fillna(data_list[2][column].mode()[0], inplace=True)
#************ Portion 4 ************
data_list[3] = data_list[3].drop(data_list[3].columns[7:16], axis=1)
data_list[3] = data_list[3].drop(data_list[3].columns[1:6], axis=1)

#fill mode value(眾數) in ["LTE Cat*"]
LTE_mode = data_list[3]["LTE Cat*"].mode()[0]
data_list[3]["LTE Cat*"] = data_list[3]["LTE Cat*"].fillna(LTE_mode)

data_list[3] = pd.get_dummies(data_list[3], columns=['Advanced cooling'], dummy_na=True)
data_list[3] = data_list[3].astype(int)
#combine each portion back to complete dataframe
data_merged = reduce(lambda  left,right: pd.merge(left, right, how='inner', left_index=True, right_index=True), data_list)

#TODO: drop rows and save [result].csv;
print(data_merged.shape)
data_merged.to_csv("G:/processed.csv")
