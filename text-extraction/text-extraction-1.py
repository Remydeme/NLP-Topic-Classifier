import pandas as pd
import numpy as np





def loadData(file_path):
    df = pd.read_csv(file_path, sep='\t')
    return df




if __name__ == "__main__":
    df = loadData(file_path='./TextFiles/smsspamcallection.tsv')

