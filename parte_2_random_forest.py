import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.special import expit, logit

# Cargar datos desde el archivo Excel
file_path = 'BBDD_PD.xlsx'
datos_historicos = pd.read_excel(file_path, sheet_name='Datos históricos')
datos_prediccion = pd.read_excel(file_path, sheet_name='Escenarios proyectados')

# Previsualizar los datos
print("Datos históricos:\n", datos_historicos.head())
print("\nDatos para predicción:\n", datos_prediccion.head())


# Preprocesar datos
# Convertir fechas a datetime
for df in [datos_historicos, datos_prediccion]:
    df['FECHA'] = pd.to_datetime(df['FECHA'])


# Aplicar la transformación logit a la Tasa de incumplimiento
def safe_logit(x):
    return logit(np.clip(x, 1e-6, 1 - 1e-6))

def safe_expit(x):
    return expit(x)

datos_historicos['Logit_Tasa'] = datos_historicos['Tasa incumplimiento'].apply(logit)
datos_historicos['Tasa incumplimiento logit'] = datos_historicos['Tasa incumplimiento'].apply(safe_logit)

# Variables independientes (X) y dependiente (y)
X = datos_historicos[['Paro', 'PIB', 'Precio_vivienda', 'Mora_Adq_Vivienda']]
y = datos_historicos['Tasa incumplimiento logit']

# Dividir en conjunto de entrenamiento y validación
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo Random Forest
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Validar el modelo
y_pred_logit = modelo.predict(X_test)
y_pred = expit(y_pred_logit)
y_test_original = expit(y_test)

rmse = mean_squared_error(y_test_original, y_pred, squared=False)
r2 = r2_score(y_test_original, y_pred)
print(f"\nDesempeño del modelo:\nRMSE: {rmse:.5f}\nR2: {r2:.5f}")

# Realizar predicciones para las fechas futuras
X_prediccion = datos_prediccion[['Paro', 'PIB', 'Precio_vivienda', 'Mora_Adq_Vivienda']]
datos_prediccion['Tasa incumplimiento logit predicha'] = modelo.predict(X_prediccion)
datos_prediccion['Tasa incumplimiento predicha'] = datos_prediccion['Tasa incumplimiento logit predicha'].apply(expit)

# Visualización de resultados
plt.figure(figsize=(10, 6))
plt.plot(datos_historicos['FECHA'], datos_historicos['Tasa incumplimiento'], label='Históricos', marker='o')
plt.plot(datos_prediccion['FECHA'], datos_prediccion['Tasa incumplimiento predicha'], label='Predicción', marker='o', linestyle='--')
plt.xlabel('Fecha')
plt.ylabel('Tasa de incumplimiento')
plt.title('Predicción de Tasa de Incumplimiento')
plt.legend()
plt.grid(True)
plt.show()

# Guardar las predicciones en un archivo
output_file = 'Predicciones_Tasa_Incumplimiento_Logit_RandomForest.xlsx'
datos_prediccion.to_excel(output_file, index=False)
print(f"\nPredicciones guardadas en: {output_file}")