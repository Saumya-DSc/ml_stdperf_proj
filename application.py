from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd



application=Flask(__name__)
app=application

## importing the models pickle files
xgb_model=pickle.load(open('models/XgbCV.pkl', 'rb'))
preprocessing_model=pickle.load(open('models/preprocessor.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        gender=request.form.get('gender')
        race_ethnicity=request.form.get('race_ethnicity')
        parental_level_of_education=request.form.get('parental_level_of_education')
        lunch=request.form.get('lunch')
        test_preparation_course=request.form.get('test_preparation_course')
        reading_score=float(request.form.get('reading_score'))
        writing_score=float(request.form.get('writing_score'))
        
        data = pd.DataFrame({
            'gender': [gender],
            'race_ethnicity': [race_ethnicity],
            'parental_level_of_education': [parental_level_of_education],
            'lunch': [lunch],
            'test_preparation_course': [test_preparation_course],
            'reading_score': [reading_score],
            'writing_score': [writing_score]
        })
        new_data_scaled=preprocessing_model.transform(data)
        result=xgb_model.predict(new_data_scaled)[0]

    
        return render_template('home.html', result=round(result,2))
    
    
    else:
        return render_template('home.html') 
        
        

if __name__=="__main__":
    app.run(host="0.0.0.0", port=8080)        


