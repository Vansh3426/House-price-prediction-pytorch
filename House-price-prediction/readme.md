House Price Prediction using Deep Learning (PyTorch) 

▢ Overview :-


This project implements a deep learning–based regression model using PyTorch to predict house prices based on multiple property-related features. The goal of the project is to 
explore how neural networks perform on structured/tabular data and to analyze training behavior, validation performance, and overfitting trends.The model is trained and evaluated on
a real-world Indian house price dataset.


▢ Problem Statement :-


• Accurate house price prediction is an important problem in real estate analytics. Given a set of features describing a property (such as location-related attributes, size, and 
  amenities), the task is to predict the price of the house as a continuous value.


▢ This project focuses on :-


• Framing house price prediction as a regression problem
• Applying a feedforward neural network
• Monitoring loss, validation loss, and MAE during training


▢ Dataset :-


•  The dataset used in this project is not included in the repository due to size constraints.

📌 Dataset Source:
https://www.kaggle.com/datasets/mohamedafsal007/house-price-dataset-of-india

• Dataset Description :-

The dataset contains house listings from India
Each sample includes 16 numerical features
Target variable represents the house price
Features (high-level)
The dataset includes features related to:
Property size and layout
Location-related attributes
Availability of amenities
Other numerical indicators affecting house prices

*Note: The dataset is preprocessed and split into training, validation, and test sets within the code.*


▢ Approach


• Model Architecture :-

Fully connected feedforward neural network
Implemented using PyTorch
Designed for regression on tabular data

• Training Strategy

Loss Function: Mean Squared Error (MSE)
Evaluation Metric: Mean Absolute Error (MAE)
Optimizer: Adam
Training performed on a local machine (GPU)

• Data Split

The dataset is split as follows (as observed during runtime):
Training set: 9356 samples
Validation set: 2340 samples
Test set: 2924 samples
Each sample contains 16 input features.

• Training Results

Below are selected training logs showing loss, validation loss, and MAE progression:

Epoch 0:
Train Loss: 0.418
Val Loss: 0.408
MAE: 0.778

Epoch 300:
Train Loss: 0.140
Val Loss: 0.145
MAE: 0.432

Epoch 600:
Train Loss: 0.125
Val Loss: 0.141
MAE: 0.425

Epoch 900:
Train Loss: 0.108
Val Loss: 0.144
MAE: 0.429

Epoch 1800:
Train Loss: 0.071
Val Loss: 0.170
MAE: 0.468


▢ Observations


• Training loss consistently decreases with more epochs
• Validation loss improves initially but starts increasing after a point
• MAE shows a U-shaped trend, indicating overfitting
• Best performance is achieved around 500–700 epochs

• This behavior highlights the importance of:

Early stopping
Regularization
Careful epoch selection


▢ Project Structure :-


House-price-prediction-pytorch/
│
├── model.py # Model definition, training loop, and evaluation
├── prediction.py # Model prediction file 
├── defaults.py # Default values for each feature  
├── README.md # Project documentation
├── requirements.txt # Python dependencies

For simplicity and experimentation, preprocessing, training, and evaluation logic are kept in a single script.

How to Run:-

1️⃣ Install dependencies

pip install -r requirements.txt

2️⃣ Download dataset

Download the dataset from the Kaggle link provided above
Place it in the expected path used in model.py

3️⃣ Train the model

python model.py


▢ Limitations :-


Neural networks may not outperform classical ML models on tabular data without extensive tuning
Model shows signs of overfitting after long training
No advanced regularization techniques (dropout, batch norm) are used
Dataset size and feature quality limit generalization


▢ Learnings :-


Practical experience using PyTorch for regression
Understanding loss vs validation loss behavior
Observing overfitting in deep learning models
Importance of evaluation metrics like MAE for regression tasks


▢ Future Improvements :-


Add early stopping
Compare with traditional ML models (Linear Regression, XGBoost)
Feature engineering and normalization improvements
Hyperparameter tuning


▢ Technologies Used :-


Python
PyTorch
NumPy
Pandas
Scikit-learn
