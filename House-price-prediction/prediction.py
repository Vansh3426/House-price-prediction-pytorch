import torch
import numpy as np
import pandas as pd
import joblib
from model import House_price


x_pipeline = joblib.load("House-price-prediction/saved_models/x_pipeline.joblib")
y_pipeline = joblib.load("House-price-prediction/saved_models/y_pipeline.joblib")

model = House_price(20)

model.load_state_dict(torch.load('House-price-prediction/saved_models/trained_model_03.pth'))

model.eval()

def prediction(model ,X ,x_pipeline ,y_pipeline):
    
    # X = np.array(X).reshape(1,-1)
    X =pd.DataFrame([X] ,columns =['number of bedrooms', 'number of bathrooms', 'living area', 'lot area',
       'number of floors', 'waterfront present', 'number of views',
       'condition of the house', 'grade of the house',
       'Area of the house(excluding basement)', 'Area of the basement',
       'Built Year', 'Renovation Year', 'Postal Code', 'Lattitude',
       'Longitude', 'living_area_renov', 'lot_area_renov',
       'Number of schools nearby', 'Distance from the airport'])
    
    # print(X.shape)

    X_scaled =x_pipeline.transform(X)
    X_tensor = torch.tensor(X_scaled , dtype=torch.float32)
    
    pred = model(X_tensor)
   
    pred =y_pipeline.inverse_transform(pred.detach().numpy()).squeeze()
    # pred = np.expm1(pred)
    pred = pred/100000
    print(f"{pred : .2f} lakhs")
    
    return pred
    
X = [3,1,900,4770,1,0,3,6,900,0,1969,2,55]
X1=[3,1.75,1820,3140,2,0,5,8,1820,0,1949,1,55]
X2 =[4,2.5,3310,42998,2,0,3,9,3310,0,2001,3,76]
X3 =[3,1.0,900,4770,1.0,0,0,3,6,900,0,1969,2009,122018,52.5338,-114.552,900,3480,2,55]
X4 =[4,1.0,1030,6621,1.0,0,0,4,6,1030,0,1955,0,122042,52.7157,-114.411,1420,6631,3,54]
output =prediction(model,X4 ,x_pipeline,y_pipeline)



