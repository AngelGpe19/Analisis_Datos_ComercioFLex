import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

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

# División de datos en entrenamiento y prueba
X = df_clean[['Edad_num', 'Frecuencia_num']].values
y = df_clean['Usar plataforma'].replace({'Si': 1, 'No': 0}).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelo de regresión logística
model_logreg = LogisticRegression()
model_logreg.fit(X_train, y_train)

# Bootstrap para estimación de intervalos de confianza
n_bootstraps = 1000
bootstrap_auc_scores = []
for _ in range(n_bootstraps):
    X_resampled, y_resampled = resample(X_train, y_train, random_state=_)
    model_logreg.fit(X_resampled, y_resampled)
    y_pred_prob = model_logreg.predict_proba(X_test)[:, 1]
    bootstrap_auc_scores.append(roc_auc_score(y_test, y_pred_prob))

# Estimación de intervalos de confianza
bootstrap_auc_scores = np.array(bootstrap_auc_scores)
confidence_interval = np.percentile(bootstrap_auc_scores, [2.5, 97.5])
print(f"Intervalo de confianza del AUC: {confidence_interval}")

# Curva ROC y AUC
y_pred_prob = model_logreg.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Comparación de modelos (Decision Tree, SVM, Random Forest)
models = {
    'Logistic Regression': model_logreg,
    'Decision Tree': DecisionTreeClassifier(),
    'SVM': SVC(probability=True),
    'Random Forest': RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Modelo: {name}")
    print(classification_report(y_test, y_pred))
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))
    print()

# Comparación de métricas adicionales
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', step='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Curva Precision-Recall')
plt.show()
