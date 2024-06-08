from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
cors = CORS(app)

# Load the machine learning model and data
model = pickle.load(open('RandomForestRegressor.pkl', 'rb'))
car_data = pd.read_csv("Cleaned_Data_For_MachineLearing.csv")


@app.route('/', methods=['GET', 'POST'])
def index():
    brands = sorted(car_data['Brand'].unique())
    years = sorted(car_data['Year'].unique(), reverse=True)
    titles = sorted(car_data['Title'].unique())
    used_or_new_options = sorted(car_data['UsedOrNew'].unique())
    transmission_options = sorted(car_data['Transmission'].unique())
    fuel_type_options = sorted(car_data['FuelType'].unique())
    cylindersinengine_type_options = sorted(car_data['CylindersinEngine'].unique())
    body_type_options = sorted(car_data['BodyType'].unique())
    color_options = sorted(car_data['Colour'].unique())

    brands.insert(0, 'Select Brand')
    years.insert(0, 'Select Year')
    used_or_new_options.insert(0, 'Select Condition')
    transmission_options.insert(0, 'Select Transmission')
    fuel_type_options.insert(0, 'Select FuelType')
    cylindersinengine_type_options.insert(0, 'Select Cylinder')
    body_type_options.insert(0, 'Select BodyType')
    color_options.insert(0, 'Select Colour')
    return render_template('index.html',
                           brands=brands,
                           years=years,
                           titles=titles,
                           used_or_new_options=used_or_new_options,
                           transmission_options=transmission_options,
                           fuel_type_options=fuel_type_options,
                           cylindersinengine_type_options=cylindersinengine_type_options,
                           body_type_options=body_type_options,
                           color_options=color_options)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    brand = request.form.get('Brand')
    title = request.form.get('Title')
    year = request.form.get('Year')
    status = request.form.get('UsedOrNew')
    transmission = request.form.get('Transmission')
    fuel_type = request.form.get('FuelType')
    cylinder = request.form.get('Cylinder')
    kilometres = request.form.get('Kilometres')
    bodytype = request.form.get('BodyType')
    colour = request.form.get('Colour')


    # Create a DataFrame with the encoded values
    input_data = pd.DataFrame(columns=['Brand', 'Year', 'Title', 'UsedOrNew', 'Transmission', 'FuelType', 'Kilometres', 'CylindersinEngine', 'BodyType', 'Colour'],
                              data=np.array([brand, year, title, status, transmission, fuel_type, kilometres, cylinder, bodytype, colour]).reshape(1, 10))

    prediction = model.predict(input_data)
    return str(np.round(prediction[0], 2))

if __name__ == '__main__':
    app.run(debug=True)
