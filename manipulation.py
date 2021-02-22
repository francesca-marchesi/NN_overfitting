import pandas as pd
import argparse

def del_missing(dataframe, col_name = None):
    if col_name == None:
        for column in dataframe:
            if dataframe[column].isnull().any() == True:
                dataframe.drop(column, axis=1, inplace= True) #this first part eliminates all the column that have mv
                #print(column)
    else:
        for name in col_name:
            dataframe = dataframe.drop(dataframe[(dataframe[name].notna() == False)].index) #eliminate the raws that have mv

    return dataframe

def import_data():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path che contiene le immagini, con / finale', type=str)
    args = parser.parse_args()
    print(args.path)
    data = pd.read_excel(args.path)
    return data
