# MiamiHousing
Miami Housing Market Sales Price Estimater

This Streamlit web application predicts housing prices in Miami using pre-trained Linear Regression and Random Forest models. Users can input property features, receive price predictions, and view data insights.

Features

- Predicts sale prices using two models: Linear Regression and Random Forest
- Accepts user input for key housing features such as land area, age, location, and structure quality
- Provides model performance metrics: R², MAE, RMSE
- Includes data visualizations:
  - Histogram of sale prices
  - Correlation heatmap
  - Scatter plots of top predictive features

How to Run

1. Clone the repository
   git clone https://github.com/yourusername/miami-housing-estimator.git
   cd miami-housing-estimator

2. Install dependencies
   pip install -r requirements.txt

3. Add the following files to the project root directory
   - linear_model.pkl
   - random_forest_model.pkl
   - scaler.pkl
   - miami_housing_sample.csv

4. Run the app
   streamlit run app.py

Project Structure

miami-housing-estimator/

housingstreamlit_py.py     # Main Streamlit application
miami_housing_sample.csv   # Sample housing dataset (not included in repo)
linear_model.pkl           # Trained linear regression model
random_forest_model.pkl    # Trained random forest model
scaler.pkl                 # StandardScaler used during model training
requirements.txt           # Python dependencies
README.md                  # Project documentation

Libraries Used

- Python
- Streamlit
- Scikit-learn
- Pandas and NumPy
- Matplotlib and Seaborn

Notes

- Latitude and longitude inputs are limited to boundaries within Miami.
- Structure quality accepts integer values from 1 to 5.
- Predictions are generated only when the "Predict" button is clicked.

License

This project was developed for educational and academic use.
Author: Shane Viola
