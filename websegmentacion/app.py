from flask import Flask, request, render_template
import pickle

# Importar los modelos
kmeans_model = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
scaler = pickle.load(open('standscaler.pkl', 'rb'))

def prediction(G,A,Ai,SS):
  features = ([[G,A,Ai,SS]])
  transformed_features = encoder.transform(features)
  transformed_features[:,2:] = scaler.transform(transformed_features[:,2:])
  prediction = kmeans_model.predict(transformed_features).reshape(1,-1)
  return prediction[0]

# Crear Flask
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    genero = request.form['Genero']
    edad = request.form['Edad']
    ingresos_anuales = request.form['Ingresos_Anuales']
    score = request.form['Score']

    # Predecir el cluster
    cluster = prediction(genero, edad, ingresos_anuales, score)

    return render_template('index.html', result="El cluster al que pertenece es {} ".format(cluster[0]))


# Python main
if __name__ == "__main__":
    app.run(debug=True, port=8080, host="0.0.0.0")

