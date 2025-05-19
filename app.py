from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))
 

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        fixed_acidity = float(request.form['fixed_acidity'])
        volatile_acidity = float(request.form['volatile_acidity'])
        citric_acid = float(request.form['citric_acid'])
        residual_sugar = float(request.form['residual_sugar'])
        chlorides  = float(request.form['chlorides'])
        free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
        total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
        density = float(request.form['density'])
        pH  = float(request.form['pH'])
        sulphates  = float(request.form['sulphates'])
        alcohol  = float(request.form['alcohol'])
        
        # Convert to numpy array
        features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
                              chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, 
                              pH, sulphates, alcohol]])
        
        

        # Make prediction
        prediction = model.predict(features)
        result = "Purchased" if prediction[0] == 1 else "Did not purchased"
    except Exception as e:
        result = f"Error in prediction: {str(e)}"
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
