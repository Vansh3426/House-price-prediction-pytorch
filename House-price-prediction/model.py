import torch
from torch import nn
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.manual_seed(42)
torch.manual_seed(42)

Xtrain= torch.load('House-price-prediction/saved_tensors/Xtrain.pt').to(device)
Xval= torch.load('House-price-prediction/saved_tensors/Xval.pt').to(device)
ytrain= torch.load('House-price-prediction/saved_tensors/ytrain.pt').to(device)
yval= torch.load('House-price-prediction/saved_tensors/yval.pt').to(device)

class House_price(nn.Module):
    
    def __init__(self ,input_feat):
        super().__init__()
        
        self.layer =nn.Sequential(
            nn.Linear(input_feat,64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,1),
            
        )
    
    def forward(self,x):
        
        pred = self.layer(x)
        
        return pred


if __name__ == "__main__" :
        
   
    # print(Xtrain.shape  ,   ytrain.shape)



    model = House_price(20).to(device)

    loss_fn = torch.nn.MSELoss()
    optimizers = torch.optim.AdamW(params=model.parameters() ,lr=0.001 ,weight_decay=0.00001 )
    
    model.train()
    best_loss = float('inf')
    epochs = 350

    loss_list =[]
    
    val_loss_list = []
    for epoch in range(epochs):
        
        pred = model(Xtrain)
        
        loss =loss_fn(pred,ytrain)
        loss_list.append(loss.item())
        optimizers.zero_grad()
        
        loss.backward()
        
        optimizers.step()
       
        
            
            
                
        model.eval()

        with torch.inference_mode():
            
            val_pred = model(Xval)
            
            val_loss =loss_fn(val_pred , yval)
            val_loss_list.append(val_loss.item())
            
            mae = torch.mean(torch.abs(yval - val_pred))
            mae = mae.item()
            # rmse = torch.sqrt(loss).item()

            
            # print(f' val loss :{loss}')
            # print(f' val MAE loss :{MAE}')
            # print(f' val loss(rmse) :{rmse}')
            
            
            
        
        if epoch % 10 == 0:
            print(f" Epochs : {epoch}    loss : {loss}     val loss :{val_loss} val MAE loss :{mae}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(),"House-price-prediction/saved_models/trained_model_03.pth")
           
    plt.plot(loss_list, label ='Train loss')
    plt.plot(val_loss_list ,label='Val loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()