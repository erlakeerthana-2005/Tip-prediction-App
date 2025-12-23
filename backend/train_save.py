""" the purpose of this file is to load the data from load_data.py
and load the trained model from train_model.py
and save the trained model to a file using joblib"""
import joblib  #to save the model
from data.load_data import load_data #to load data
from model.train_model import train_model #to train model

def train_and_save_model():
 """load data ,train the model and save the trained model to a file"""
#load the data
df=load_data()

#train the model
model=train_model(df)

#save the trained model to a file
joblib.dump(model,'random_forest_model.pkl')
print("model trained and saved to random_forest_model.pkl")

if __name__=="__main__":
    train_and_save_model()  