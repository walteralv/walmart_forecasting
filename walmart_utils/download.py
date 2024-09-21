import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt 
import plotly.io as pio
import plotly.express as px
import plotly.graph_objs as go
from tqdm import tqdm

import os 
from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



def extract_dataset(zip_path: str, extract_dir: str):
    if not os.path.exists(zip_path):
        print(f"El archivo {zip_path} no existe.")
        return None
    if not os.path.exists(extract_dir):
        print(f"La carpeta de destino {extract_dir} no existe. Creando la carpeta... 🫡")
        os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Archivo extraído en: {extract_dir} 🫣")


# Para los archivos zip
def download_dataset():
    def format(df, columna_date):
        df[columna_date] = pd.to_datetime(df[columna_date], yearfirst=True)
        #df.set_index(columna_date, inplace=True)
        df.sort_values(columna_date, ascending=True, inplace=True)
        return df
    try:
        zip_path = "./walmart-recruiting-store-sales-forecasting.zip"
        extract_dir = "./walmart_dataset"
        extract_dataset(zip_path , extract_dir)
        zip_path = "./walmart_dataset/features.csv.zip"
        extract_dir = "./walmart_dataset"
        extract_dataset(zip_path , extract_dir)
        zip_path = "./walmart_dataset/sampleSubmission.csv.zip"
        extract_dir = "./walmart_dataset"
        extract_dataset(zip_path , extract_dir)
        zip_path = "./walmart_dataset/test.csv.zip"
        extract_dir = "./walmart_dataset"
        extract_dataset(zip_path , extract_dir)
        zip_path = "./walmart_dataset/train.csv.zip"
        extract_dir = "./walmart_dataset"
        extract_dataset(zip_path , extract_dir)

        #Para dataframes
        features_df = pd.read_csv("./walmart_dataset/features.csv", date_format=False)
        stores_df = pd.read_csv("./walmart_dataset/stores.csv", date_format=False)
        test_df = pd.read_csv("./walmart_dataset/test.csv", date_format=False)
        train_df = pd.read_csv("./walmart_dataset/train.csv", date_format=False)
        print("dataset descargado con éxito :)")
        train_df = format(train_df, "Date")
        test_df = format(test_df, "Date")
        features_df = format(features_df, "Date")
        test_df["Weekly_Sales"] = 0
        print("dataset formateado :)")
        return features_df, stores_df, test_df, train_df
    except Exception as e:
        print("no s epudo descargar :(")
        return None, None, None, None
    

