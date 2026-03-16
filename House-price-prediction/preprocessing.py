import torch
import pandas as pd
import joblib
import pandas as pd 
import joblib
from  sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler ,PowerTransformer
from sklearn.pipeline import Pipeline

# df = pd.read_csv('House-price-prediction/data/House Price India.csv')
# df =df.drop(columns=['id','Date'])
# print(df.columns)

# print(df['Price'].describe())

# percentile_25 =df['Price'].quantile(0.25)
# percentile_75 =df['Price'].quantile(0.75)

# IQR = percentile_75 -percentile_25

# lower_limit = percentile_25 - 1.5 * IQR
# upper_limit = percentile_75 + 1.5 * IQR

# print(percentile_25)
# print(percentile_75)
# print(IQR)
# print(upper_limit)
# print(lower_limit)

# new_df = df[ df['Price'] < upper_limit]
# new_df = df[ df['Price'] > lower_limit]

# print(new_df.count())

# # Save the DataFrame to a CSV file
# new_df.to_csv('House-price-prediction/data/cleaned_dataset.csv', index=False) # index=False prevents pandas from writing row indices as a column

if __name__ == '__main__':
    
    # df = pd.read_csv('House-price-prediction/data/House Price India.csv')
    # df =df.drop(columns=['id','Date'])
    df = pd.read_csv('House-price-prediction/data/cleaned_dataset.csv')
    inputs=df.drop(columns=['Price'])
    print(inputs.shape)
    target =df['Price']
    print(df.count)
    print(df.columns)
    
    
    
    x_pipeline = Pipeline([
    ('power', PowerTransformer(method='yeo-johnson')),
    ('scale', StandardScaler())
    ])
    
    y_pipeline = Pipeline([
    ('power', PowerTransformer(method='yeo-johnson')),  
    ('scale', StandardScaler())                          
    ])
    
    Xtrain ,Xtest ,ytrain ,ytest = train_test_split(inputs,target,test_size=0.2,shuffle=True,random_state=42)
    Xtrain ,Xval , ytrain ,yval =train_test_split(Xtrain ,ytrain ,test_size=0.2, shuffle=True ,random_state=42)
    
    print( Xtrain.shape  ,   Xval.shape   , Xtest.shape)
    
                
                
                
                
                
                
                
                
                
                
        
    Xtrain =x_pipeline.fit_transform(Xtrain)
    Xval =x_pipeline.transform(Xval)
    Xtest =x_pipeline.transform(Xtest)


    ytrain = ytrain.to_numpy().reshape(-1,1)
    yval = yval.to_numpy().reshape(-1,1)
    ytest = ytest.to_numpy().reshape(-1,1)
   

    ytrain =y_pipeline.fit_transform(ytrain)
    yval =y_pipeline.transform(yval)
    ytest =y_pipeline.transform(ytest)
    


    joblib.dump(x_pipeline ,"House-price-prediction/saved_models/x_pipeline.joblib")
    joblib.dump(y_pipeline ,"House-price-prediction/saved_models/y_pipeline.joblib")
    
    
    Xtrain = torch.tensor(Xtrain ,dtype=torch.float32 )
    Xval = torch.tensor(Xval ,dtype=torch.float32)
    Xtest = torch.tensor(Xtest ,dtype=torch.float32)
    ytrain = torch.tensor(ytrain ,dtype=torch.float32)
    yval = torch.tensor(yval ,dtype=torch.float32 )
    ytest = torch.tensor(ytest ,dtype=torch.float32)

    torch.save(Xtrain ,'House-price-prediction/saved_tensors/Xtrain.pt')
    
    torch.save(Xval ,'House-price-prediction/saved_tensors/Xval.pt')
    
    torch.save(Xtest ,'House-price-prediction/saved_tensors/Xtest.pt')
    
    torch.save(ytrain ,'House-price-prediction/saved_tensors/ytrain.pt')
    
    torch.save(yval ,'House-price-prediction/saved_tensors/yval.pt')
    
    torch.save(ytest ,'House-price-prediction/saved_tensors/ytest.pt')