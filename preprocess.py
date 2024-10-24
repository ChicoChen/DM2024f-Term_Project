import pandas as pd
from functools import reduce
import numpy as np

#read csv and seperate it as a list of length 4;
PATH = "/data/nanoreview_phone_specs (processed).csv"
data = pd.read_csv("./data/nanoreview_phone_specs (processed).csv")
data_list= [data.iloc[:, i:i+27] for i in range(0, 108, 27)]

#TODO: process each portion of data
#************ Portion 1 ************

#************ Portion 2 ************

#************ Portion 3 ************

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
data_merged.to_csv("./data/processed.csv")
