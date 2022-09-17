from flask import Flask, jsonify, request, render_template, redirect, url_for, flash, send_file
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
import pickle
import pandas as pd
import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

#os.chdir(os.path.dirname(__file__))

# Creamos la aplicación
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

# Cargamos el modelo de machine learning que será usado para las predicciones
model = pickle.load(open(f'{os.getcwd()}/my_model.pkl', 'rb'))


# creamos una clase para que el usuario pueda subir el archivo CSV que será utilizado para crear la predicción:
class UploadFileForm(FlaskForm):
    file = FileField('File', validators=[InputRequired()])
    submit = SubmitField('Upload File')

# Creamos la aplicación, página principal con el buscador de archivos para que el usuario pueda subir el csv
@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def index():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))
        
        #return 'El archivo se ha cargado correctamente'
        #data = []
        #with open(file) as file:
        #    csvfile = csv.reader(file)
        #    for row in csvfile:
        #        data.append(row)
        #    data = pd.DataFrame(data)
    return render_template('home.html', form=form)

# creamos la página de predicción
@app.route("/predict", methods=['GET','POST'])
def predict():
    contenido_carpeta = os.listdir(f'{os.getcwd()}/static/files')
    file_ = pd.read_csv(f'{os.getcwd()}/static/files/{contenido_carpeta[0]}')
    df = file_.drop(['Date_of_termination', 'Date_of_Hire', 'Source_of_Hire', 'Higher_Education', 'Work_accident', 'Department', 'Mode_of_work', 'JobInvolvement', 'JobSatisfaction', 'Absenteeism', 'PerformanceRating','YearsAtCompany', 'MonthlyIncome', 'NumCompaniesWorked'], axis=1)

    rename_jobrole = {
        'Laboratory Technician': 'Lab_Technician', 
        'Research Scientist': 'Scientist',
        'Research Director': 'Director',
        'Sales Representative': 'Sales_Rep',
        'Sales Executive': 'Sales_Executive',
        'Manager': 'Manager',
        'Manufacturing Director': 'Director',
        'Healthcare Representative': 'Healthcare_Rep',
        'Human Resources': 'HR'
    }

    df.replace(rename_jobrole, inplace=True)

    df = pd.get_dummies(df,columns=['BusinessTravel',
    'JobRole',
    'Gender',
    'MaritalStatus',
    'OverTime',
    'Job_mode',
    'Status_of_leaving'], 
    drop_first=True)
    
    prediction = model.predict(df)

    rename_prediccion = {
        1: 'Renuncia',
        0: 'No Renuncia'
    }
    
    df_result = pd.DataFrame(prediction)
    df_result.replace(rename_prediccion, inplace=True)
    
    df_final = file_
    df_final['attrition'] = df_result

    df_final.to_csv(f'{os.getcwd()}/static/downloads/resultado.csv')

    return download()

    #for i in prediction:
    #    if i == True:
    #        resultado = 'Renuncia'
            
    #    else:
    #        resultado = 'No renuncia'
            
    
    #tamano = len(df)

    #if tamano>1:
    #    tabla = pd.DataFrame(df,columns=df.columns)
    #    tabla['prediction'] = prediction
    #    resultado = ['Renuncia' if p==True else 'No Renuncia' for p in prediction ]
    #        #tabla_html=tabla.to_html()
    #    return render_template('predict_table.html',tabla=tabla.to_html())
    #elif tamano==1:
    #    return render_template('predict.html',resultado=prediction[0])


    #return jsonify ({'predictions': result})
        
    #return jsonify ({'predictions': str(prediction)})
    #return render_template('home.html', resultado=resultado)

@app.route('/predict/download')
def download():
    contenido_carpeta_ = os.listdir(f'{os.getcwd()}/static/downloads')
    path = f'{os.getcwd()}/static/downloads/{contenido_carpeta_[0]}'
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug = True)