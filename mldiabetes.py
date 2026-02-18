import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Cargar dataset
df = pd.read_csv("diabetes_risk_dataset.csv")

# Separar features y target
X = df.drop(["Patient_ID","diabetes_risk_category"], axis=1)
y = df["diabetes_risk_category"]

# Codificar variables categóricas (ej. gender, physical_activity_level, family_history_diabetes)
for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entrenar modelo
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Guardar modelo y columnas
joblib.dump(model, "modelo_diabetes.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "columnas.pkl")
