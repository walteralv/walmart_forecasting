import pandas as pd

import matplotlib.pyplot as plt 
import plotly.io as pio
import plotly.express as px
import plotly.graph_objs as go

def view_data_horizon(
        data_df: pd.DataFrame, store: int= 1, 
        dept: int= 1, target: str="Weekly_Sales", 
        back_horizon: int= 52, 
    ):
    plot_df = data_df[(data_df["Store"] == store) & (data_df["Dept"] == dept)].copy()
    plot_df[f"lag_{back_horizon}_{target}"] = plot_df.groupby(["Store", "Dept"])[target].shift(back_horizon)
    plot_df.set_index("Date", inplace=True)
    plot_df.sort_values("Date", ascending=True, inplace=True)
    fig = px.line(
        plot_df, 
        y=[f"{target}", f"lag_{back_horizon}_{target}"], 
        title=f"{target} and Lag {back_horizon} Weeks",
    )
    fig.show()
    del plot_df