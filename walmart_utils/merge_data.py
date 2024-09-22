import pandas as pd

def dedupe(df_list_dd):
    for i in df_list_dd:
        i.drop_duplicates(inplace=True)



def merge_data(train_df, test_df, stores_df, features_df):
    data = pd.concat([train_df, test_df])
    data = data.merge(stores_df, how="inner", on="Store")
    data = data.merge(features_df, how="inner", on= ["Store", "Date"])
    data.sort_values(by=["Store", "Dept", "Date"], ascending=True, inplace=True)
    return data