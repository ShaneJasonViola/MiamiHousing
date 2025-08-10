Miami Housing Price Estimator
================================

A Streamlit app that predicts Miami home sale prices using two pre-trained models
(Linear Regression and Random Forest). Enter property features, get an estimated price,
and explore quick visual insights from the sample dataset.

---

Features
--------
- Price predictions from **Linear Regression** and **Random Forest**
- Clean sidebar inputs for property details (land area, living area, distances, age, etc.)
- Model performance metrics: **R²**, **MAE**, **RMSE**
- Visuals:
  - Histogram of sale prices
  - Correlation heatmap
  - Scatter plots for the most predictive features
- Light, Miami-themed UI (custom CSS)

---

Quick Start (Local)
-------------------
1) Clone this repo:
   git clone https://github.com/yourusername/miami-housing-estimator.git
   cd miami-housing-estimator

2) Install dependencies:
   pip install -r requirements.txt

3) Add these files to the project root (not committed to the repo):
   - linear_model.pkl
   - random_forest_model.pkl
   - scaler.pkl
   - miami_housing_sample.csv

4) Run the app:
   streamlit run app.py

   The app will open in your browser (usually http://localhost:8501).

---

Run on Google Colab
-------------------
You can also launch the app from Colab:
1) Open a new Colab notebook.
2) In the first cell, install requirements and launch Streamlit:

   %%bash
   pip -q install streamlit scikit-learn pandas numpy matplotlib seaborn joblib pyngrok

3) Upload your files (click the folder icon in the left panel and use the upload button):
   - app.py
   - miami_housing_sample.csv
   - linear_model.pkl
   - random_forest_model.pkl
   - scaler.pkl

4) In a new cell, run the app via a tunnel (simple Ngrok approach):

   import nest_asyncio, threading, socket, os, time
   from pyngrok import ngrok

   # Kill any existing tunnels
   for t in ngrok.get_tunnels():
       ngrok.disconnect(t.public_url)

   # Start Streamlit on a free port
   port = 8501
   public_url = ngrok.connect(port).public_url
   print("Public URL:", public_url)

   def run_streamlit():
       os.system(f"streamlit run app.py --server.port {port} --server.address 0.0.0.0")

   thread = threading.Thread(target=run_streamlit, daemon=True)
   thread.start()

   # Give Streamlit a moment to boot
   time.sleep(5)
   print("App is starting... Open the Public URL above.")

5) Click the printed **Public URL** to use your Streamlit app.

Notes:
- If you see a “ModuleNotFoundError”, re-run the install cell.
- If a model/scaler file is missing, the app will stop with a clear error message.

---

Project Structure
-----------------
miami-housing-estimator/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation (this file's content)
├── miami_bg.jpg            # Optional background image for the Miami theme
├── miami_housing_sample.csv  # Sample dataset (add locally; not in repo)
├── linear_model.pkl        # Trained Linear Regression model (add locally)
├── random_forest_model.pkl # Trained Random Forest model (add locally)
└── scaler.pkl              # Fitted StandardScaler (add locally)

Required Files (not in repo)
----------------------------
- linear_model.pkl
- random_forest_model.pkl
- scaler.pkl
- miami_housing_sample.csv

These are expected in the same folder as app.py at runtime.

---

Troubleshooting
---------------
- File not found: Double‑check the model files and dataset are in the project root.
- Port already in use: Close any other Streamlit sessions or change the port in the Colab snippet.
- White text on white background: This project ships custom CSS; if styling looks off,
  clear the Streamlit cache and refresh (or remove your custom theme settings from ~/.streamlit).

---

License
-------
This project is for educational use.
Author: Shane Viola
