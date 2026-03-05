import torch
import numpy as np
import pandas as pd
import joblib
from model import House_price


x_scaler = joblib.load("House-price-prediction/saved_models/x_scaler.joblib")
y_scaler = joblib.load("House-price-prediction/saved_models/y_scaler.joblib")

model = House_price(16)

model.load_state_dict(torch.load('House-price-prediction/saved_models/trained_model_02.pth'))

model.eval()

def prediction(model ,X ,x_scaler ,y_scaler):
    
    # X = np.array(X).reshape(1,-1)
    X =pd.DataFrame([X] ,columns =["number of bedrooms",'number of bathrooms','living area','lot area','number of floors','waterfront present','condition of the house','grade of the house','Area of the house(excluding basement)','Area of the basement','Built Year','Renovation Year','living_area_renov','lot_area_renov','Number of schools nearby','Distance from the airport'])
    
    # print(X.shape)

    X_scaled =x_scaler.transform(X)
    X_tensor = torch.tensor(X_scaled , dtype=torch.float32)
    
    pred = model(X_tensor)
   
    pred =y_scaler.inverse_transform(pred.detach().numpy()).squeeze()
    pred = np.expm1(pred)
    pred = pred/100000
    print(f"{pred : .2f} lakhs")
    
    return pred
    
X = [3,1,900,4770,1,0,3,6,900,0,1969,2009,900,3480,2,55]
X1=[3,1.75,1820,3140,2,0,5,8,1820,0,1949,1990,2030,5499,1,55]
X2 =[4,2.5,3310,42998,2,0,3,9,3310,0,2001,0,3350,42847,3,76]

output =prediction(model,X2 ,x_scaler,y_scaler)



