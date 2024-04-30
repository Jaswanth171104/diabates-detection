from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__,template_folder='templates')

with open('modeln.pkl', 'rb') as m:
    model = pickle.load(m)
# Your machine learning model training code and necessary imports

@app.route('/',methods=['GET','POST'])


def index():
    if request.method == 'POST':
        Pregnancies= float(request.form['Pregnancies'])
        Glucose = float(request.form['Glucose'])
        BloodPressure= float(request.form['BloodPressure'])
        SkinThickness = float(request.form['SkinThickness'])
        Insulin = float(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabatesPedigreeFunction = float(request.form['DiabatesPedigreeFunction'])
        Age=float(request.form['Age'])
       

    # Repeat for other features

    # Make prediction
        input_data = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'] # Use the received data
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        prediction = model.predict(input_data_reshaped)

    # Interpret prediction
        if (prediction[0] == 0):
            result='The person is not diabetic'
        else:
            result='The person is diabetic'

        return render_template('index.html', result=result)
        
    return render_template('index.html')

    # Retrieve form data
    

if __name__ == "__main__":
    app.run(debug=True) 
