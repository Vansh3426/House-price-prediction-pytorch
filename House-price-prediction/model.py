import torch
from torch import nn
import numpy as np
import pandas as pd 
import joblib
from  sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.manual_seed(44)
torch.manual_seed(42)

class House_price(nn.Module):
    
    def __init__(self ,input_feat):
        super().__init__()
        
        self.layer =nn.Sequential(
            nn.Linear(input_feat,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,1),
            
        )
    
    def forward(self,x):
        
        pred = self.layer(x)
        
        return pred


if __name__ == "__main__" :
        
    df = pd.read_csv('House-price-prediction/data/House Price India.csv')

    inputs =df.drop(columns =['id','Date','Price','Postal Code','Lattitude','Longitude','number of views'])
  
    df['Price'] = np.log1p(df['Price'])
    target =df['Price']
    # print(target.max())
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()


    Xtrain ,Xtest ,ytrain ,ytest = train_test_split(inputs,target,test_size=0.2,shuffle=True,random_state=42)
    Xtrain ,Xval , ytrain ,yval =train_test_split(Xtrain ,ytrain ,test_size=0.2 ,random_state=42)
    
    print( Xtrain.shape  ,   Xval.shape   , Xtest.shape)
    
    Xtrain =x_scaler.fit_transform(Xtrain)
    Xval =x_scaler.transform(Xval)
    Xtest =x_scaler.transform(Xtest)


    ytrain = ytrain.to_numpy().reshape(-1,1)
    yval = yval.to_numpy().reshape(-1,1)
    ytest = ytest.to_numpy().reshape(-1,1)
    # print(Xtrain.shape , ytrain.shape)

    ytrain =y_scaler.fit_transform(ytrain)
    yval =y_scaler.transform(yval)
    ytest =y_scaler.transform(ytest)


    joblib.dump(x_scaler ,"House-price-prediction/saved_models/x_scaler.joblib")
    joblib.dump(y_scaler ,"House-price-prediction/saved_models/y_scaler.joblib")


    Xtrain = torch.tensor(Xtrain ,dtype=torch.float32 ,device=device)
    Xval = torch.tensor(Xval ,dtype=torch.float32 ,device=device)
    Xtest = torch.tensor(Xtest ,dtype=torch.float32 ,device=device)
    ytrain = torch.tensor(ytrain ,dtype=torch.float32,device=device)
    yval = torch.tensor(yval ,dtype=torch.float32 ,device=device)
    ytest = torch.tensor(ytest ,dtype=torch.float32 ,device=device)



    # print(Xtrain.shape  ,   ytrain.shape)



    model = House_price(16).to(device)

    loss_fn = torch.nn.SmoothL1Loss()
    optimizers = torch.optim.AdamW(params=model.parameters() ,lr=0.0001 , weight_decay=0.0001)

    eval_loss_fn = torch.nn.L1Loss()
    
    model.train()
    best_loss = float('inf')
    epochs = 1850

    for epoch in range(epochs):
        
        pred = model(Xtrain)
        
        loss =loss_fn(pred,ytrain)
        
        optimizers.zero_grad()
        
        loss.backward()
        
        optimizers.step()
       
        
            
            
                
        model.eval()

        with torch.inference_mode():
            
            val_pred = model(Xval)
            
            val_loss =loss_fn(val_pred , yval)
            MAE = eval_loss_fn(val_pred ,yval)
            # rmse = torch.sqrt(loss).item()

            
            # print(f' val loss :{loss}')
            # print(f' val MAE loss :{MAE}')
            # print(f' val loss(rmse) :{rmse}')
            
            
            
        
        if epoch % 50 == 0:
            print(f" Epochs : {epoch}    loss : {loss}     val loss :{val_loss}     MAE : {MAE}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(),"House-price-prediction/saved_models/trained_model_01.pth")
           
        