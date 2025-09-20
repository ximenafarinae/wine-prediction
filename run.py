import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data.data_loader import load_data
from src.data.data_transformer import normalize_input, remove_outliers_iqr
from src.models.model import NeuralNetwork
from src.config import *
from src.models.trainer import train
from src.utils.data.utils import *

data = load_data(DATA_PATH)
print("antes de limpiar: ", len(data))
# Cambio los valores de la columna 'quality' a 0 o 1 según si es menor igual a 5
data['quality'] = (data['quality'] >= 5).astype('int8')

# Filtrás las filas con quality == 1 y te quedás con 6000
df_positivos = data[data['quality'] == 1].sample(n=6000, random_state=42)

# Filtrás las filas con quality == 0 y también recortás a 6000
df_negativos = data[data['quality'] == 0].sample(n=6000, random_state=42)

# Unís ambos subconjuntos
df_balanceado = pd.concat([df_positivos, df_negativos], axis=0)

# Mezclás el orden de las filas (importante para entrenar)
df_balanceado = df_balanceado.sample(frac=1, random_state=42).reset_index(drop=True)


# Tomo las columnas relevantes (segun matriz de correlacion)
relevant_columns = df_balanceado.iloc[:, :-1]

# Normalizo las columnas de entrada.
norm_data = normalize_input(relevant_columns)


# Elimino los valores atípicos usando la técnica rango intercuartilico (IQR)
cleaned_data = remove_outliers_iqr(norm_data, norm_data.columns)

# Elimino todas las columnas que contengas datos NaN
print("despues de normalizar y limpiar: ", len(cleaned_data))


# Grafico la distribucion de los datos
#data_distribution(norm_data)

# Datos de entrada
all_inputs = cleaned_data.values

# Datos de salida
all_outputs = df_balanceado[['quality']].values

# Dividir en un conjunto de entrenamiento y uno de prueba
X_train, X_test, Y_train, Y_test = train_test_split(all_inputs, all_outputs, test_size=1/3)
n = X_train.shape[0]
m = X_test.shape[0]
print("Cantidad de datos de entrenamiento: ", len(X_train))
print("Cantidad de datos de prueba: ", len(X_test))

print("Distribución de clases en entrenamiento: ")
print(pd.Series(Y_train.flatten()).value_counts())
print("Distribución de clases en validación: ")
print(pd.Series(Y_test.flatten()).value_counts())

# Se inicializa el modelo
model = NeuralNetwork(INPUT_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_SIZE)

# Entrenamiento de la red
#train_accuracies, val_accuracies, train_losses, val_losses = train(np, model, X_train, Y_train, X_test, Y_test, EPOCHS, L, n, m)

# Grafica de perdida y precision
#plot_accuracy_and_loss(train_accuracies, val_accuracies, train_losses, val_losses)
EXPERIMENTS = [
    dict(name="exp_baseline",   HIDDEN_LAYER_SIZE=11, L=1e-5,  EPOCHS=300),
    dict(name="exp_small_lr",   HIDDEN_LAYER_SIZE=11, L=5e-6,  EPOCHS=3000),
    dict(name="exp_big_lr",     HIDDEN_LAYER_SIZE=11, L=5e-5,  EPOCHS=3000),
    dict(name="exp_more_hidden",HIDDEN_LAYER_SIZE=16, L=1e-5,  EPOCHS=3000),
    dict(name="exp_less_hidden",HIDDEN_LAYER_SIZE=6,  L=1e-5,  EPOCHS=3000),
    dict(name="h4_lr1e-4_e4000",   HIDDEN_LAYER_SIZE=4,  L=1e-4,  EPOCHS=400),
    dict(name="h6_lr3e-4_e3000",   HIDDEN_LAYER_SIZE=6,  L=3e-4,  EPOCHS=3000),
    dict(name="h8_lr6e-4_e2500",   HIDDEN_LAYER_SIZE=8,  L=6e-4,  EPOCHS=2500),
    dict(name="h10_lr1e-3_e1800",  HIDDEN_LAYER_SIZE=10, L=1e-3,  EPOCHS=180),
    dict(name="h12_lr8e-4_e2200",  HIDDEN_LAYER_SIZE=12, L=8e-4,  EPOCHS=2200),
    dict(name="h14_lr5e-4_e3000",  HIDDEN_LAYER_SIZE=14, L=5e-4,  EPOCHS=3000),
    dict(name="h16_lr7e-4_e2600",  HIDDEN_LAYER_SIZE=16, L=7e-4,  EPOCHS=2600),
    dict(name="h18_lr3e-4_e3500",  HIDDEN_LAYER_SIZE=18, L=3e-4,  EPOCHS=350),
    dict(name="h20_lr2e-4_e4000",  HIDDEN_LAYER_SIZE=20, L=2e-4,  EPOCHS=4000),
    dict(name="h24_lr1e-4_e4500",  HIDDEN_LAYER_SIZE=24, L=1e-4,  EPOCHS=4500),
    dict(name="h6_lr5e-4_e400",    HIDDEN_LAYER_SIZE=6,   L=5e-4,   EPOCHS=400),
    dict(name="h8_lr7e-4_e350",    HIDDEN_LAYER_SIZE=8,   L=7e-4,   EPOCHS=350),
    dict(name="h10_lr1e-3_e300",   HIDDEN_LAYER_SIZE=10,  L=1e-3,   EPOCHS=300),
    dict(name="h12_lr8e-4_e500",   HIDDEN_LAYER_SIZE=12,  L=8e-4,   EPOCHS=500),
    dict(name="h16_lr6e-4_e600",   HIDDEN_LAYER_SIZE=16,  L=6e-4,   EPOCHS=600),
    dict(name="h20_lr9e-4_e550",   HIDDEN_LAYER_SIZE=20,  L=9e-4,   EPOCHS=550),
    dict(name="h24_lr5e-4_e700",   HIDDEN_LAYER_SIZE=24,  L=5e-4,   EPOCHS=700),
    dict(name="h32_lr6e-4_e800",   HIDDEN_LAYER_SIZE=32,  L=6e-4,   EPOCHS=800),
    dict(name="h14_lr1e-3_e450",   HIDDEN_LAYER_SIZE=14,  L=1e-3,   EPOCHS=450),
    dict(name="h18_lr1e-3_e650",   HIDDEN_LAYER_SIZE=18,  L=1e-3,   EPOCHS=650),
    dict(name="h28_lr5e-4_e750",   HIDDEN_LAYER_SIZE=28,  L=5e-4,   EPOCHS=750),
    dict(name="h8_lr1e-3_e250",    HIDDEN_LAYER_SIZE=8,   L=1e-3,   EPOCHS=250),
    dict(name="h4_lr1e-3_e200",    HIDDEN_LAYER_SIZE=4,   L=1e-3,   EPOCHS=200),
    dict(name="h6_lr2e-3_e180",    HIDDEN_LAYER_SIZE=6,   L=2e-3,   EPOCHS=180),
    dict(name="h12_lr2e-3_e200",   HIDDEN_LAYER_SIZE=12,  L=2e-3,   EPOCHS=200),
    dict(name="h16_lr2_5e-3_e180", HIDDEN_LAYER_SIZE=16,  L=2.5e-3, EPOCHS=180),
    dict(name="h8_lr3e-3_e150",    HIDDEN_LAYER_SIZE=8,   L=3e-3,   EPOCHS=150),
    dict(name="h10_lr1_5e-3_e220", HIDDEN_LAYER_SIZE=10,  L=1.5e-3, EPOCHS=220),
    dict(name="h14_lr1_2e-3_e240", HIDDEN_LAYER_SIZE=14,  L=1.2e-3, EPOCHS=240),
    dict(name="h20_lr8e-4_e400",   HIDDEN_LAYER_SIZE=20,  L=8e-4,   EPOCHS=400),
    dict(name="h24_lr1_1e-3_e420", HIDDEN_LAYER_SIZE=24,  L=1.1e-3, EPOCHS=420),
    dict(name="h32_lr9e-4_e600",   HIDDEN_LAYER_SIZE=32,  L=9e-4,   EPOCHS=600),
    dict(name="h6_lr7e-4_e300",    HIDDEN_LAYER_SIZE=6,   L=7e-4,   EPOCHS=300),
    dict(name="h18_lr6e-4_e500",   HIDDEN_LAYER_SIZE=18,  L=6e-4,   EPOCHS=500),

    dict(name="h6_lr1e-3_e1500",   HIDDEN_LAYER_SIZE=6,  L=1e-3,  EPOCHS=1500),
    dict(name="h8_lr9e-4_e1800",   HIDDEN_LAYER_SIZE=8,  L=9e-4,  EPOCHS=1800),
    dict(name="h10_lr6e-4_e2400",  HIDDEN_LAYER_SIZE=10, L=6e-4,  EPOCHS=2400),
    dict(name="h12_lr4e-4_e3000",  HIDDEN_LAYER_SIZE=12, L=4e-4,  EPOCHS=3000),
    dict(name="h14_lr3e-4_e3600",  HIDDEN_LAYER_SIZE=14, L=3e-4,  EPOCHS=3600),
    dict(name="h16_lr2e-4_e4200",  HIDDEN_LAYER_SIZE=16, L=2e-4,  EPOCHS=4200),
    dict(name="h18_lr1e-4_e5000",  HIDDEN_LAYER_SIZE=18, L=1e-4,  EPOCHS=5000),
    dict(name="h20_lr8e-4_e2000",  HIDDEN_LAYER_SIZE=20, L=8e-4,  EPOCHS=200),
    dict(name="h22_lr5e-4_e2600",  HIDDEN_LAYER_SIZE=22, L=5e-4,  EPOCHS=2600),
    dict(name="h24_lr3e-4_e3200",  HIDDEN_LAYER_SIZE=24, L=3e-4,  EPOCHS=3200),
    dict(name="h3_lr5e-4_e600",     HIDDEN_LAYER_SIZE=3,  L=5e-4,  EPOCHS=600),
    dict(name="h3_lr1e-3_e400",     HIDDEN_LAYER_SIZE=3,  L=1e-3,  EPOCHS=400),
    dict(name="h3_lr2e-3_e300",     HIDDEN_LAYER_SIZE=3,  L=2e-3,  EPOCHS=300),

    dict(name="h4_lr5e-4_e800",     HIDDEN_LAYER_SIZE=4,  L=5e-4,  EPOCHS=800),
    dict(name="h4_lr1e-3_e600",     HIDDEN_LAYER_SIZE=4,  L=1e-3,  EPOCHS=600),
    dict(name="h4_lr2e-3_e450",     HIDDEN_LAYER_SIZE=4,  L=2e-3,  EPOCHS=450),

    dict(name="h5_lr3e-4_e1000",    HIDDEN_LAYER_SIZE=5,  L=3e-4,  EPOCHS=1000),
    dict(name="h5_lr7e-4_e800",     HIDDEN_LAYER_SIZE=5,  L=7e-4,  EPOCHS=800),
    dict(name="h5_lr1e-3_e600",     HIDDEN_LAYER_SIZE=5,  L=1e-3,  EPOCHS=600),

    dict(name="h6_lr2e-4_e1200",    HIDDEN_LAYER_SIZE=6,  L=2e-4,  EPOCHS=1200),
    dict(name="h6_lr5e-4_e900",     HIDDEN_LAYER_SIZE=6,  L=5e-4,  EPOCHS=900),
    dict(name="h6_lr1e-3_e650",     HIDDEN_LAYER_SIZE=6,  L=1e-3,  EPOCHS=650),

    dict(name="h7_lr5e-4_e900",     HIDDEN_LAYER_SIZE=7,  L=5e-4,  EPOCHS=900),
    dict(name="h7_lr1e-3_e700",     HIDDEN_LAYER_SIZE=7,  L=1e-3,  EPOCHS=700),
    dict(name="h7_lr2e-3_e500",     HIDDEN_LAYER_SIZE=7,  L=2e-3,  EPOCHS=500),

    dict(name="h8_lr3e-4_e1000",    HIDDEN_LAYER_SIZE=8,  L=3e-4,  EPOCHS=1000),
    dict(name="h8_lr7e-4_e800",     HIDDEN_LAYER_SIZE=8,  L=7e-4,  EPOCHS=800),
    dict(name="h8_lr1e-3_e600",     HIDDEN_LAYER_SIZE=8,  L=1e-3,  EPOCHS=600),

    dict(name="h9_lr5e-4_e850",     HIDDEN_LAYER_SIZE=9,  L=5e-4,  EPOCHS=850),
    dict(name="h9_lr1e-3_e650",     HIDDEN_LAYER_SIZE=9,  L=1e-3,  EPOCHS=650),
    dict(name="h9_lr2e-3_e450",     HIDDEN_LAYER_SIZE=9,  L=2e-3,  EPOCHS=450),

    dict(name="h11_lr5e-4_e800",    HIDDEN_LAYER_SIZE=11, L=5e-4,  EPOCHS=800),
    dict(name="h11_lr1e-3_e600",    HIDDEN_LAYER_SIZE=11, L=1e-3,  EPOCHS=600),
    dict(name="h11_lr2e-3_e400",    HIDDEN_LAYER_SIZE=11, L=2e-3,  EPOCHS=400),

    dict(name="h16_lr3e-4_e1000",   HIDDEN_LAYER_SIZE=16, L=3e-4,  EPOCHS=1000),
    dict(name="h16_lr7e-4_e800",    HIDDEN_LAYER_SIZE=16, L=7e-4,  EPOCHS=800),
    dict(name="h16_lr1e-3_e600",    HIDDEN_LAYER_SIZE=16, L=1e-3,  EPOCHS=600),

    dict(name="h24_lr3e-4_e900",    HIDDEN_LAYER_SIZE=24, L=3e-4,  EPOCHS=900),
    dict(name="h24_lr7e-4_e700",    HIDDEN_LAYER_SIZE=24, L=7e-4,  EPOCHS=700),
    dict(name="h24_lr1e-3_e500",    HIDDEN_LAYER_SIZE=24, L=1e-3,  EPOCHS=500),

    dict(name="h32_lr2e-4_e1000",   HIDDEN_LAYER_SIZE=32, L=2e-4,  EPOCHS=1000),
    dict(name="h32_lr5e-4_e800",    HIDDEN_LAYER_SIZE=32, L=5e-4,  EPOCHS=800),
    dict(name="h32_lr1e-3_e600",    HIDDEN_LAYER_SIZE=32, L=1e-3,  EPOCHS=600),
]

# === correr experimentos ===
results = {}

for cfg in EXPERIMENTS:
    print(f"\n=== Entrenando {cfg['name']} ===")

    HIDDEN_LAYER_SIZE = cfg['HIDDEN_LAYER_SIZE']
    L = cfg['L']
    EPOCHS = cfg['EPOCHS']

    model = NeuralNetwork(INPUT_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_SIZE)

    train_acc, val_acc, train_loss, val_loss = train(
        np, model, X_train, Y_train, X_test, Y_test, EPOCHS, L, n, m
    )

    results[cfg['name']] = {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "train_loss": train_loss,
        "val_loss": val_loss
    }

    # podés graficar para cada experimento
    plot_accuracy_and_loss(train_acc, val_acc, train_loss, val_loss, title=cfg['name'])

# al final podés comparar resultados
for name, res in results.items():
    print(f"{name}: final val_acc={res['val_acc'][-1]:.3f}, val_loss={res['val_loss'][-1]:.3f}")
