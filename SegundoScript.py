import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Cargar datos de Excel
df = pd.read_excel('D:\Sexto cuatri\Tesis\Respuestas de la encuesta.xlsx')

# Corregir nombres de columnas
df.columns = [
    'Genero', 'Edad', 'Ocupacion', 'Actividad economica', 'Frecuencia adquisicion',
    'Aspecto importante', 'Posee aparatos', 'Redes sociales', 'Usar plataforma',
    'Utilizar plataforma', 'Categorias productos', 'Tipos productos', 'Motivacion plataforma',
    'Participar subastas', 'Forma entrega', 'Comprar con descuento', 'Anuncios publicitarios',
    'Medio preferido', 'Anuncios plataforma', 'Suscripcion', 'Paquete preferido',
    'Pago paquete estandar', 'Pago paquete premium', 'Medios de pago', 'Promociones preferidas',
    'Informacion plataforma', 'Ubicacion oficinas', 'Suscripcion anual'
]

# Mapeo de respuestas a valores numéricos
edad_map = {
    'a) Menor de 18 años': 1,
    'b) Entre 18 a 25 años': 2,
    'c) Entre 26 a 35 años': 3,
    'd) Entre 36 a 45 años': 4,
    'e) Entre 46 a 55 años': 5,
    'f) Entre 56 a 65 años': 6
}

frecuencia_map = {
    'a) De 1 a 3 veces a la semana': 1,
    'b) De 4 a 6 días en la semana': 2,
    'c) Todos los días': 3,
    'd) Dos veces al mes': 4,
    'e) De dos a tres veces al mes': 5,
    'f) Cada 6 meses': 6,
    'g) Cada año': 7,
    'h) Nunca': 8
}

df['Edad_num'] = df['Edad'].map(edad_map)
df['Frecuencia_num'] = df['Frecuencia adquisicion'].map(frecuencia_map)

# Filtrar filas con NaN
df_clean = df.dropna(subset=['Edad_num', 'Frecuencia_num'])

# Estadísticas descriptivas
print("Estadísticas Descriptivas:")
print(df_clean[['Edad_num', 'Frecuencia_num']].describe())

# Gráficos de distribución y campana de Gauss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df_clean['Edad_num'], kde=True, bins=6, stat='density')
plt.title('Distribución de Edad')
plt.xlabel('Edad')
plt.ylabel('Densidad')

plt.subplot(1, 2, 2)
sns.histplot(df_clean['Frecuencia_num'], kde=True, bins=8, stat='density')
plt.title('Distribución de Frecuencia de Adquisición')
plt.xlabel('Frecuencia de Adquisición')
plt.ylabel('Densidad')

plt.tight_layout()
plt.show()

# Análisis de correlación
correlation_matrix = df_clean[['Edad_num', 'Frecuencia_num']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matriz de Correlación')
plt.show()

# Modelo de Regresión Logística
X = df_clean[['Edad_num', 'Frecuencia_num']].copy()
y = df_clean['Usar plataforma'].replace({'Si': 1, 'No': 0})

# Escalar las variables numéricas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Búsqueda de hiperparámetros
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Mejores hiperparámetros encontrados:", best_params)

# Predicciones
y_pred = best_model.predict(X_test)

# Evaluación del modelo
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

# Curva ROC y AUC
y_proba = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='Curva ROC')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Línea Base')
plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
plt.title('Curva ROC')
plt.legend()
plt.show()

# AUC
auc = roc_auc_score(y_test, y_proba)
print(f"\nÁrea bajo la curva (AUC): {auc}")
