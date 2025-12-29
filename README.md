# 💰 Customer Revenue Prediction (Neural Network + Streamlit App)

## 📌 Overview
This project is an interactive **Streamlit web application** that predicts customer revenue based on website engagement behavior using a **pre-trained Deep Neural Network model** built with **TensorFlow/Keras**.

The model estimates revenue in **log scale** and converts it into actual dollar value for business interpretation.  
It was trained using **train/validation/test split**, ensuring no future data leakage during training.

## 🎯 Objective
- Predict customer monetization potential based on session activity.
- Provide both **individual and batch revenue forecasting**.
- Segment customers into revenue groups for business insights.
- Deploy a machine learning model in an **easy-to-use demo interface**.

## ✨ Key Features
- Manual input or **CSV upload for batch prediction**
- Automatic encoding for `deviceCategory` if needed
- Input validation with clear expected feature format
- Revenue conversion from **log → actual USD**
- Business metrics display:
  - Average Revenue
  - Max Revenue
  - Total Predicted Revenue
  - Customer Segmentation (Low / Medium / High)
- Downloadable results after prediction
- Cached model loading for optimized performance

## 🧠 Model Information
| Component | Description |
|---|---|
| Model Type | Deep Neural Network (Regression) |
| Framework | TensorFlow / Keras |
| Input Features | 8 numerical features |
| Output | Log Revenue (1 neuron) |
| Training Method | Train / Validation / Test Split |
| Optimization | Early Stopping + Hyperparameter Tuning |
| Feature Scaling | Robust numerical scaling applied during training |

## 📊 Input Feature Format (Required for CSV Upload)
Your dataset must contain the following 8 columns **in exact order**

### Suggested realistic value ranges:
- `deviceCategory_encoded`: 0 = Desktop, 1 = Mobile, 2 = Tablet
- `month`: 1 → 12
- `year`: 2016 → 2017
- `hits`: 0 → 500 (average around 4)
- `pageviews`: 0 → 469 (average around 3.35)
- `mean_hits_month`: 0 → 500
- `mean_pageviews_month`: 0 → 466
- `visitNumber`: 1 → 395 (average around 1.98)

> Note: All inputs should be numeric or convertible to `float32`.

## 🛠 Tech Stack
- **Frontend**: Streamlit
- **Model**: TensorFlow / Keras
- **Data Processing**: Pandas, NumPy
- **Model Deployment**: `.h5` model file
- **Caching**: `st.cache_resource`, `st.cache_resource`
- **Math & Transformation**: Log output converted using `np.exp()`

### 4. Make Predictions
- **Manual Input**: Fill customer data in the interface and click **Predict Revenue**
- **CSV Upload**: Upload a valid CSV file and click **Predict All Revenue**

## 📈 Business Interpretation
| Revenue Level | Meaning |
|---|---|
| < $1 | Low monetization potential |
| $1 → $10 | Medium potential |
| > $10 | High monetization potential |
