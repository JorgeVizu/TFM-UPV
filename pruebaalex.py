import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import shapiro, kstest

# Cargar los datos desde un archivo xlsx
data = pd.read_excel("datos_final_sin_minimo_keyword.xlsx")

X_train = pd.read_csv("x_train.csv", index_col=False)
X_test = pd.read_csv("x_test.csv", index_col=False)
y_train = pd.read_csv("y_train.csv", index_col=False)
y_test = pd.read_csv("y_test.csv", index_col=False)

print(X_train)