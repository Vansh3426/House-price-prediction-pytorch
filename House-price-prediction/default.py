import pandas as pd
import joblib

df = pd.read_csv('House-price-prediction/data/House Price India.csv')

defaults ={
"number_of_bedrooms": df["number of bedrooms"].median(),
"number_of_bathrooms": df["number of bathrooms"].median(),
"number_of_floors": df["number of floors"].median(),
"number_of_schools_nearby": df["Number of schools nearby"].median(),
"living_area": df["living area"].median(),
"lot_area": df["lot area"].median(),
"area_no_basement": df["Area of the house(excluding basement)"].median(),
"basement": df["Area of the basement"].median(),
"living_area_renov": df["living_area_renov"].median(),
"lot_area_renov": df["lot_area_renov"].median(),
"waterfront_present": df["waterfront present"].mode()[0],
"condition_of_the_house": df["condition of the house"].mode()[0],
"grade_of_the_house": df["grade of the house"].mode()[0],
"built_year": df["Built Year"].median(),
"renovation_year": df["Renovation Year"].median(),
"distance_from_airport": df["Distance from the airport"].median(),
}

joblib.dump(defaults, "House-price-prediction/saved_models/defaults.joblib")
