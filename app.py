from flask import Flask, render_template, request, jsonify
import pickle 
import numpy as np

app = Flask(__name__)

# Loading GRADIENT BOOSTING model
with open("static/gb_model.pkl", "rb") as f:
    gb_model = pickle.load(f)

@app.route("/")
def home():
    return  render_template('index.htm')

@app.route("/", methods=["POST"])
def predict():
    
# Fetching all values by user
    BHK = request.form['BHK']
    cr_ar = request.form['cr_ar']
    Balcony = request.form['Balcony']
    Furnishing = request.form.get('Furnishing')
    Floor = request.form['Floor']
    garden = request.form.get('garden')
    Main_Road = request.form.get('Main_Road')
    Pool = request.form.get('Pool')

# Predicting the amount of house
    features = [f"{BHK}", f"{cr_ar}", f"{Balcony}", f"{Furnishing}", f"{Floor}", f"{garden}", f"{Main_Road}", f"{Pool}"]
    values = [float(x) for x in features]
    X = np.array([values])
    pred = gb_model.predict(X)

    return render_template('index.htm',pred=int(pred[0]))

if __name__ == "__main__":
    app.run(debug=True)