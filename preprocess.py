import pandas as pd
import numpy as np

#read csv and seperate it as a list of length 4;
PATH = "/data/nanoreview_phone_specs (processed).csv"
data = pd.read_csv(".`/data/nanoreview_phone_specs (processed).csv")
data_list= [data.iloc[:, i:i+27] for i in range(0, 108, 27)]
print(data_list[3].columns)

#TODO: process each portion of data


#TODO: combine each portion back to complete dataframe


#TODO: drop rows and save [result].csv;

