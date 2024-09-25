import pandas as pd
from tqdm import tqdm

def add_calendar_features(
        data: pd.DataFrame,  
        columns_id: list[str]= ["Store", "Dept", "Date"],
    ) -> pd.DataFrame:
    data["month"] = data["Date"].dt.month 
    data["day_of_month"] = data["Date"].dt.days_in_month
    #data["day_of_week"] = data["Date"].dt.day_of_week
    data["week_of_year"] = data["Date"].dt.isocalendar().week 
    data["quarter"] = data["Date"].dt.quarter
    data["year"] = data["Date"].dt.year
    data.sort_values(by=columns_id, ascending=True, inplace=True)
    return data 


def sum_markdown(
        data: pd.DataFrame,
        colums: list[str] = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4","MarkDown5"],
        columns_id: list[str]= ["Store", "Dept", "Date"],
    ) -> pd.DataFrame:

    data["MarkDown"] = data[colums].sum(axis=1)
    data.drop(colums, axis=1, inplace=True)
    data.sort_values(by=columns_id, ascending=True, inplace=True)
    return data


def add_holiday(
    data: pd.DataFrame, 
    holiday_list: list[str],
    columns_id: list[str]= ["Store", "Dept", "Date"],    
) -> pd.DataFrame:
    holiday_list = pd.to_datetime(holiday_list, format='%d-%m-%y', yearfirst=True)
    data["IsHoliday_2"] = data["Date"].isin(holiday_list).astype(int)
    data["IsHoliday"] = data["IsHoliday_2"] | data["IsHoliday"]
    data.drop("IsHoliday_2", axis=1, inplace=True)
    data.sort_values(by=columns_id, ascending=True, inplace=True)
    return data
    

def add_memory_feature(
        data_df: pd.DataFrame,
        target: str= "Weekly_Sales",
        columns_id: list[str]= ["Store", "Dept", "Date"],
        back_horizon: int= 52,
        lags: list[int]= [1,2,3], # USO LA TARGET DE HACE 52 SEMANAS, 53, 54
        aggregation_windows: list[int]= [3,4] #  USO UN GRUPO DE TARGET DE HACE 52 SEMANAS [3,4, 5]
    ) -> pd.DataFrame:
    #data = data_df[columns_id + [target,]].copy()
    data = data_df.copy()
    data.sort_values(by=columns_id, ascending=True, inplace=True)
    lags = [0, ] + lags
    for lag in tqdm(lags, desc="Adding memory variable 😸"):
        memory = back_horizon + lag 
        data[f"lag_{lag}_{target}"] = data.groupby(["Store", "Dept"])[target].shift(memory)
        for w in tqdm(aggregation_windows,desc="Adding window aggregation 🫡"):
            data[f"l_{lag}_w_{w}_sum_{target}"] = data.groupby(["Store", "Dept"])[target].shift(memory).rolling(window=w).sum()
            data[f"l_{lag}_w_{w}_mean_{target}"] = data.groupby(["Store", "Dept"])[target].shift(memory).rolling(window=w).mean()
            data[f"l_{lag}_w_{w}_median_{target}"] = data.groupby(["Store", "Dept"])[target].shift(memory).rolling(window=w).median()
            data[f"l_{lag}_w_{w}_std_{target}"] = data.groupby(["Store", "Dept"])[target].shift(memory).rolling(window=w).std()
            data[f"l_{lag}_w_{w}_max_{target}"] = data.groupby(["Store", "Dept"])[target].shift(memory).rolling(window=w).max()
            data[f"l_{lag}_w_{w}_min_{target}"] = data.groupby(["Store", "Dept"])[target].shift(memory).rolling(window=w).min()
    data.sort_values(by=columns_id, ascending=True, inplace=True)
    return data

# join con fechas pasadas 