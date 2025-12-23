import streamlit as st #to create the web app
import pandas as pd #to handle dataframe
import requests #to make api calls 

st.title("Tip  Prediction App") #title of the app
st.write("Please enter the details to get the tip amount.") #description of the app

#input data
total_bill=st.number_input("Total Bill Amount",min_value=0.0)
sex=st.selectbox("Sex",options=['Male','Female'])
smoker=st.selectbox("Smoker",options=['Yes','No'])          
day=st.selectbox("Day",options=['Thur','Fri','Sat','Sun'])
time=st.selectbox("Time",options=['Lunch','Dinner'])    
size=st.number_input("Size of the party",min_value=1,max_value=10)
if st.button("Predict "):
    #prepare the data to be sent to the backend 
    input_data={'total_bill':total_bill,
                'sex':sex,
                'smoker':smoker,    
                'day':day,
                'time':time,
                'size':size}
    
    response=requests.post('http://127.0.0.1:5000/predict',json=input_data) #make the api call to the backend
    if response.status_code==200:
        prediction=response.json().get('prediction') #get the predicted tip amount from the response
        st.write('The tip value is ',prediction) #display the predicted  amount
    else:
        st.write('Error in prediction')
        