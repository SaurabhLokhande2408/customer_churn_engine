from pathlib import Path
import pandas as pd

def data_loading():
    data_path = Path(__file__).resolve().parent / "data" / "telco.csv"
    data = pd.read_csv(data_path)
    return data
