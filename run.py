import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data.data_loader import load_data
from src.data.data_transformer import normalize_input, remove_outliers_percentile
from src.models.model import NeuralNetwork
from src.config import *
from src.models.trainer import train
from src.utils.data.utils import *

data = load_data(DATA_PATH)

# Elimino todas las columnas que contengas datos NaN
data = data.dropna(axis=1, how='all')

# Cambio los valores de la columna 'quality' a 0 o 1 según si es menor igual a 5
data['quality'] = data['quality'].apply(lambda x: 1 if x >= 5 else 0)

# Tomo las columnas relevantes (segun matriz de correlacion)
relevant_columns = data.values[:2, -1]

print("col relevantes", data.columns)

# Elimino los valores atípicos usando la técnica de percentile ya que mis datos tienen una fuerte asimetria hacia la derecha
data = remove_outliers_percentile(data, data.columns)

# Normalizo las columnas de entrada.
norm_data = normalize_input(data)
# relevant_columns = data.columns.dropna(axis=0, how='any')

# Grafico la distribucion de los datos
# data_distribution(norm_data)

# Datos de entrada
all_inputs = norm_data.values

# Datos de salida
all_outputs = norm_data[["quality"]].values

# Dividir en un conjunto de entrenamiento y uno de prueba
X_train, X_test, Y_train, Y_test = train_test_split(all_inputs, all_outputs,
    test_size=1/3)
n = X_train.shape[0]
m = X_test.shape[0]
print("Cantidad de datos de entrenamiento: ", len(X_train))
print("Cantidad de datos de prueba: ", len(X_test))

print("Distribución de clases en entrenamiento:")
print(pd.Series(Y_train.flatten()).value_counts())
print("Distribución de clases en validación:")
print(pd.Series(Y_test.flatten()).value_counts())

# Se inicializa el modelo
model = NeuralNetwork(INPUT_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_SIZE)

# Entrenamiento de la red
train_accuracies, val_accuracies, train_losses, val_losses = train(np, model, X_train, Y_train, X_test, Y_test, EPOCHS, L, n, m)

# Grafica de perdida y precision
plot_accuracy_and_loss(train_accuracies, val_accuracies, train_losses, val_losses)
