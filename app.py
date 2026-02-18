import io
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, render_template, send_file
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar dataset y modelo
df = pd.read_csv("diabetes_risk_dataset.csv")
model = joblib.load("modelo_diabetes.pkl")
scaler = joblib.load("scaler.pkl")
columnas = joblib.load("columnas.pkl")

@app.route("/")
def home():
    return render_template("index.html", columnas=columnas)

@app.route("/predict", methods=["POST"])
def predict():
    valores = [float(request.form[col]) for col in columnas]
    valores_scaled = scaler.transform([valores])
    prediccion = model.predict(valores_scaled)[0]
    return render_template("index.html", columnas=columnas, prediction_text=f"Categoría de riesgo: {prediccion}")

# Nueva ruta para ver gráficas
@app.route("/graficas")
def graficas():
    # Ejemplo: distribución de BMI por categoría de riesgo
    plt.figure(figsize=(8,6))
    sns.boxplot(x="diabetes_risk_category", y="bmi", data=df, palette="Set2")
    plt.title("Distribución de BMI por categoría de riesgo")

    # Guardar en memoria como PNG
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plt.close()
    return send_file(img, mimetype="image/png")

# Otra ruta para ver clusters con KMeans
@app.route("/clusters")
def clusters():
    from sklearn.cluster import KMeans

    # Usar las mismas columnas que el scaler
    X = df.drop(["Patient_ID","diabetes_risk_category"], axis=1)
    X_scaled = scaler.transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=df["bmi"], y=df["fasting_glucose_level"], hue=df["cluster"], palette="tab10")
    plt.title("Clusters de pacientes por BMI y Glucosa en ayunas")

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plt.close()
    return send_file(img, mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True)
