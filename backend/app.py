from flask import Flask, request, jsonify
#flask to create an api end point
#request to get data from the streamlit app
#jsonify  is to convert the data into json format and return it to the streamlit app
import joblib #to load the trained model
import pandas as pd #to handle dataframe

app=Flask(__name__) #create a flask app or create an api end point

#load the trained model
model=joblib.load('random_forest_model.pkl')

@app.route('/predict',methods=['POST'])
def predict():
    """api end point to get data from the streamlit app 
    and return the predicted tip amount"""

    data = request.get_json() #get the data from the streamlit app
    #convert the data into a pandas dataframe

    input_df =pd.DataFrame([{'total_bill':data['total_bill'],
                            'sex':data['sex'],
                            'smoker':data['smoker'],
                            'day':data['day'],
                            'time':data['time'],
                            'size':data['size']}]) #convert the data into a  dataframe
    
    #make the prediction
    prediction=model.predict(input_df) #make the prediction
    return jsonify({'prediction':prediction[0]}) #return the prediction to the streamlit app

if __name__=="__main__":
    app.run(debug=True) #run the app in debug mode
