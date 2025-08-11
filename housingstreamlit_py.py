# app.py — Miami Housing Price Estimator

import base64
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Miami Housing Price Estimator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# MIAMI THEME (CSS)
# -----------------------------------------------------------------------------
def apply_miami_theme(bg_path: str = "miami_bg.jpg"):
    """
    Miami-styled theme using custom CSS:
    - optional background image
    - gradient H1, neon accents
    - frosted sidebar
    - styled buttons/inputs
    - dark readable text across the main app
    """
    b64 = ""
    p = Path(bg_path)
    if p.exists():
        try:
            b64 = base64.b64encode(p.read_bytes()).decode()
        except Exception:
            b64 = ""

    gradient = "linear-gradient(135deg, #2bd2ff 0%, #ff6ec7 50%, #7a2ee6 100%)"

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Poppins', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
        color: #0e1726;
    }}

    .stApp {{
        background: {"url('data:image/jpeg;base64," + b64 + "') no-repeat center/cover fixed" if b64 else "linear-gradient(180deg, #e9fbff 0%, #fff0ff 100%)"};
    }}
    .stApp:before {{
        content:"";
        position: fixed; inset: 0;
        background: rgba(255,255,255,0.72);
        z-index: 0;
    }}
    .block-container {{ position: relative; z-index: 1; }}

    /* Sidebar (frosted) */
    section[data-testid="stSidebar"] > div {{
        background: rgba(255,255,255,0.88);
        backdrop-filter: blur(8px);
        border-right: 1px solid rgba(0,0,0,0.06);
    }}

    /* Headers */
    h1 {{
        background: {gradient};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        letter-spacing: .2px;
        margin-top: .2rem;
    }}

    /* Buttons */
    .stButton>button {{
        background: {gradient};
        color: #fff; border: 0; border-radius: 12px;
        padding: .6rem 1.1rem; font-weight: 600;
        box-shadow: 0 6px 18px rgba(122,46,230,.25);
        transition: transform .06s ease, box-shadow .12s ease;
    }}
    .stButton>button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 10px 26px rgba(122,46,230,.35);
    }}

    /* Inputs */
    .stSelectbox, .stNumberInput, .stTextInput, .stDateInput, .stMultiSelect {{
        background: rgba(255,255,255,0.92);
        border-radius: 10px !important;
    }}
    div[data-baseweb="select"] > div, input[type="number"], input[type="text"] {{
        border-radius: 10px !important;
    }}

    /* Sidebar label color (bright Miami orange) */
    section[data-testid="stSidebar"] label {{
        color: #ff8c00 !important;
        font-weight: 600 !important;
    }}

    /* Cards / alerts */
    div[data-testid="stMetric"] > div, div[role="alert"] {{
        background: rgba(255,255,255,0.86);
        border-radius: 14px;
        border: 1px solid rgba(0,0,0,0.05);
        box-shadow: 0 10px 30px rgba(45,184,255,0.15);
        padding: 14px;
    }}

    /* DataFrame header */
    .stDataFrame thead tr th {{
        background: #0e1726 !important;
        color: #fff !important;
        border: 0 !important;
    }}

    /* Slider accent */
    div[data-baseweb="slider"] > div > div > div {{
        background-image: {gradient} !important;
    }}

    /* Force DARK text across main content (subheaders, paragraphs, etc.) */
    .block-container h2,
    .block-container h3,
    .block-container p,
    .block-container span,
    .block-container strong,
    .block-container label,
    div[data-testid="stMarkdownContainer"] * {{
      color: #0e1726 !important;
    }}

    /* Also ensure text inside metrics and alerts is dark */
    div[data-testid="stMetric"] *,
    div[role="alert"] * {{
      color: #0e1726 !important;
    }}

    #MainMenu, footer {{ visibility: hidden; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Apply theme and plotting palette
apply_miami_theme("miami_bg.jpg")
sns.set_style("whitegrid")
sns.set_palette(["#2bd2ff", "#ff6ec7", "#7a2ee6", "#00d8b4", "#ffd166"])

# -----------------------------------------------------------------------------
# APP TITLE / INTRO
# -----------------------------------------------------------------------------
st.title("Miami Housing Price Estimator")
st.markdown("""
This application predicts housing prices based on user input using Linear Regression and Random Forest models.
It also provides model evaluation metrics and key insights from the dataset.
""")

# -----------------------------------------------------------------------------
# LOAD MODELS AND SCALER
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        lr_model = joblib.load("linear_model.pkl")
        rf_model = joblib.load("random_forest_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return lr_model, rf_model, scaler
    except FileNotFoundError as e:
        st.error(f"Missing model file: {e.filename}. Please ensure all model files are uploaded.")
        st.stop()

lr_model, rf_model, scaler = load_models()

# -----------------------------------------------------------------------------
# LOAD DATASET
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("miami_housing_sample.csv")
        return df
    except FileNotFoundError:
        st.error("Dataset file 'miami_housing_sample.csv' not found. Please upload the dataset.")
        st.stop()

df = load_data()
feature_columns = df.drop("SALE_PRC", axis=1).columns.tolist()

# -----------------------------------------------------------------------------
# UI LABELS
# -----------------------------------------------------------------------------
feature_labels = {
    "LND_SQFOOT": "Land Area (sq ft)",
    "TOT_LVG_AREA": "Living Area (sq ft)",
    "SPEC_FEAT_VAL": "Special Feature Value ($)",
    "RAIL_DIST": "Distance to Nearest Rail Line (ft)",
    "OCEAN_DIST": "Distance to Ocean (ft)",
    "WATER_DIST": "Distance to Water Body (ft)",
    "CNTR_DIST": "Distance to Downtown Miami (ft)",
    "SUBCNTR_DI": "Distance to Subcenter (ft)",
    "HWY_DIST": "Distance to Highway (ft)",
    "age": "Age of Structure",
    "avno60plus": "Airplane Noise > 60 dB (1=Yes)",
    "structure_quality": "Structure Quality (1–5)",
    "month_sold": "Month Sold (1 = Jan)",
    "LATITUDE": "Latitude",
    "LONGITUDE": "Longitude"
}

# -----------------------------------------------------------------------------
# USER INPUTS (SIDEBAR)
# -----------------------------------------------------------------------------
st.sidebar.header("Input House Features")

def get_user_input():
    input_data = {}

    month_options = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }

    for feature in feature_columns:
        label = feature_labels.get(feature, feature)

        if feature == "month_sold":
            month_name = st.sidebar.selectbox("Month Sold", options=list(month_options.keys()))
            input_data[feature] = month_options[month_name]

        elif feature == "LATITUDE":
            input_data[feature] = st.sidebar.number_input(
                label, min_value=25.50, max_value=25.90,
                value=float(df[feature].median()), step=0.01, format="%.6f"
            )

        elif feature == "LONGITUDE":
            input_data[feature] = st.sidebar.number_input(
                label, min_value=-80.35, max_value=-80.10,
                value=float(df[feature].median()), step=0.01, format="%.6f"
            )

        elif feature == "structure_quality":
            input_data[feature] = st.sidebar.number_input(
                label, min_value=1, max_value=5,
                value=int(df[feature].median()), step=1, format="%d"
            )

        elif df[feature].dtype in [np.float64, np.int64]:
            input_data[feature] = st.sidebar.number_input(
                label, value=int(df[feature].median()), step=1, format="%d"
            )

        else:
            input_data[feature] = st.sidebar.selectbox(
                label, options=sorted(df[feature].unique())
            )

    return pd.DataFrame([input_data])

user_input_df = get_user_input()

# -----------------------------------------------------------------------------
# PREDICTION
# -----------------------------------------------------------------------------
if st.sidebar.button("Predict"):
    if user_input_df.isnull().values.any():
        st.warning("Some input fields are missing. Please complete all inputs.")
        st.stop()

    try:
        user_input_df = user_input_df[scaler.feature_names_in_]
    except AttributeError:
        st.error("Scaler does not contain feature names. It may have been trained on a NumPy array. Please retrain it on a DataFrame.")
        st.stop()
    except KeyError as e:
        st.error(f"Input features do not match the scaler's expected input. Missing columns: {e}")
        st.stop()

    user_scaled = scaler.transform(user_input_df)
    lr_pred = lr_model.predict(user_scaled)[0]
    rf_pred = rf_model.predict(user_scaled)[0]

    st.subheader("Estimated Property Price")
    st.write(f"Linear Regression Prediction: ${lr_pred:,.2f}")
    st.write(f"Random Forest Prediction: ${rf_pred:,.2f}")

    rf_test_preds = rf_model.predict(scaler.transform(df[feature_columns]))
    rf_pred_std = np.std(rf_test_preds)
    st.info(f"Approximate Confidence Range for Random Forest Prediction: ±${rf_pred_std:.2f}")
else:
    st.info("Enter values in the sidebar and click Predict to see results.")

# -----------------------------------------------------------------------------
# MODEL PERFORMANCE
# -----------------------------------------------------------------------------
st.subheader("Model Performance Comparison")

X = df[feature_columns]
y = df["SALE_PRC"]
X_scaled = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

lr_test_pred = lr_model.predict(X_test)
rf_test_pred = rf_model.predict(X_test)

def compute_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }

lr_metrics = compute_metrics(y_test, lr_test_pred)
rf_metrics = compute_metrics(y_test, rf_test_pred)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Linear Regression Metrics**")
    st.write(f"MAE : {lr_metrics['MAE']:.2f}")
    st.write(f"RMSE: {lr_metrics['RMSE']:.2f}")
    st.write(f"R²  : {lr_metrics['R2']:.4f}")

with col2:
    st.markdown("**Random Forest Metrics**")
    st.write(f"MAE : {rf_metrics['MAE']:.2f}")
    st.write(f"RMSE: {rf_metrics['RMSE']:.2f}")
    st.write(f"R²  : {rf_metrics['R2']:.4f}")

st.markdown("### Model Comparison Summary")
better_model = "Random Forest" if rf_metrics["R2"] > lr_metrics["R2"] else "Linear Regression"
st.success(f"Based on R², the better performing model is: {better_model}.")

# -----------------------------------------------------------------------------
# DATA EXPLORATION & VISUALS
# -----------------------------------------------------------------------------
st.subheader("Data Exploration and Visual Insights")

# Distribution of Sale Prices
st.markdown("#### Distribution of Sale Prices")
fig1, ax1 = plt.subplots(figsize=(12, 6))
sns.histplot(df["SALE_PRC"], bins=40, ax=ax1, kde=True, color="#2bd2ff")
ax1.set_title("Distribution of Sale Prices", fontsize=16)
ax1.set_xlabel("Sale Price ($)", fontsize=13)
ax1.set_ylabel("Home Sales", fontsize=13)
ax1.ticklabel_format(style='plain', axis='x')
ax1.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "${:,}".format(int(x))))
plt.tight_layout()
st.pyplot(fig1)

# Correlation Heatmap
st.markdown("Correlation Heatmap")
renamed_df = df.rename(columns=feature_labels)
corr_matrix = renamed_df.corr(numeric_only=True)
fig2, ax2 = plt.subplots(figsize=(6, 5))
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2,
    annot_kws={"size": 6}, cbar_kws={"shrink": 0.6}
)
ax2.tick_params(axis='x', labelrotation=90, labelsize=6)
ax2.tick_params(axis='y', labelsize=6)
st.pyplot(fig2)

# Scatter plots of top features
st.markdown("Scatter Plots of Key Predictive Features")
target_corr = df.corr(numeric_only=True)["SALE_PRC"].drop("SALE_PRC").abs().sort_values(ascending=False)
top_features = target_corr.head(4).index

fig3, axs = plt.subplots(2, 2, figsize=(15, 10))
axs = axs.flatten()
for i, feature in enumerate(top_features):
    label = feature_labels.get(feature, feature)
    sns.scatterplot(data=df, x=feature, y="SALE_PRC", ax=axs[i])
    axs[i].set_title(f"{label} vs Sale Price")
    axs[i].set_xlabel(label)
    axs[i].set_ylabel("Sale Price ($)")
plt.tight_layout()
st.pyplot(fig3)

# --- FAST MAP (PyDeck) ---
import pydeck as pdk
import numpy as np

st.subheader("Miami Home Sales Map (fast)")

needed_cols = {"LATITUDE", "LONGITUDE", "SALE_PRC"}
if not needed_cols.issubset(df.columns):
    st.info("Missing columns for map.")
else:
    # Basic filters
    pmin, pmax = int(df["SALE_PRC"].min()), int(df["SALE_PRC"].max())
    lo, hi = st.slider("Price range ($)", pmin, pmax, (pmin, pmax), step=5000)
    max_points = st.slider("Max points on map", 1000, 20000, 5000, 1000)

    @st.cache_data(show_spinner=False)
    def prep_map_data(df_in, lo, hi, max_points):
        d = df_in.dropna(subset=["LATITUDE","LONGITUDE","SALE_PRC"]).copy()
        # tighten to Miami bounding box (optional)
        d = d[
            d["LATITUDE"].between(25.40, 26.10) &
            d["LONGITUDE"].between(-80.60, -79.90)
        ]
        d = d[(d["SALE_PRC"] >= lo) & (d["SALE_PRC"] <= hi)]
        # downsample if too big
        if len(d) > max_points:
            d = d.sample(max_points, random_state=42)
        # build an orange → red color scale by price
        rng = max(1, hi - lo)
        t = ((d["SALE_PRC"] - lo) / rng).clip(0, 1).to_numpy()
        # bright orange to deeper red
        d["color_r"] = (220 + 35*t).astype(int)      # 220→255
        d["color_g"] = (120 - 80*t).clip(0,120).astype(int)  # 120→40
        d["color_b"] = 0
        return d[["LONGITUDE","LATITUDE","SALE_PRC","color_r","color_g","color_b"]]

    df_map = prep_map_data(df, lo, hi, max_points)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position="[LONGITUDE, LATITUDE]",
        get_radius=30,  # meters
        get_fill_color="[color_r, color_g, color_b, 170]",
        pickable=True,
    )
    view = pdk.ViewState(latitude=25.77, longitude=-80.19, zoom=9)
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        tooltip={"text": "Price: ${SALE_PRC}"},
        map_provider="carto",            # no Mapbox token needed
        map_style="light",
    )
    st.pydeck_chart(deck, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Developed for academic purposes only.")





