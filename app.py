import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np

# Page config
st.set_page_config(page_title="Revenue Prediction", page_icon="💰", layout="wide")

# Title
st.title("💰 Customer Revenue Prediction")
st.markdown("Predict customer revenue using neural network model")

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        # Load model without compile
        model = tf.keras.models.load_model('revenue_model.h5', compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def create_feature_vector(device_encoded, hits, mean_hits_month, 
                         mean_pageviews_month, month, pageviews, visit_number, year):
    """Create feature vector with CORRECT ORDER matching training data (8 features)"""
    
    features = np.array([
        device_encoded,          # 1. deviceCategory_encoded
        hits,                    # 2. hits
        mean_hits_month,         # 3. mean_hits_month
        mean_pageviews_month,    # 4. mean_pageviews_month
        month,                   # 5. month
        pageviews,               # 6. pageviews
        visit_number,            # 7. visitNumber
        year                     # 8. year
    ], dtype=np.float32)
    
    return features.reshape(1, -1)

# Load model
model = load_model()

if model is not None:
    st.success("✅ Model loaded successfully!")
    st.info(f"📊 Model expects {model.input_shape[1]} features")
    
    # Sidebar for input method
    st.sidebar.header("Input Method")
    input_method = st.sidebar.selectbox(
        "Choose input method:",
        ["Manual Input", "Upload CSV"]
    )
    
    if input_method == "Manual Input":
        st.subheader("Enter Customer Data")
        
        # Organize inputs in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📱 Device & Basic Info**")
            device_options = {"Desktop": 0, "Mobile": 1, "Tablet": 2}
            device_category = st.selectbox("Device Category", list(device_options.keys()))
            device_encoded = device_options[device_category]
            
            hits = st.number_input(
                "Hits", 
                min_value=0.0, 
                max_value=500.0, 
                value=4.0,
                help="Number of hits (0-500, mean: 4.0)"
            )
            
            pageviews = st.number_input(
                "Pageviews", 
                min_value=0.0, 
                max_value=469.0, 
                value=3.35,
                help="Number of pageviews (0-469, mean: 3.35)"
            )
            
            visit_number = st.number_input(
                "Visit Number", 
                min_value=0.0, 
                max_value=395.0, 
                value=1.98,
                help="Visit number (1-395, mean: 1.98)"
            )
        
        with col2:
            st.markdown("**📊 Monthly Averages & Time**")
            mean_hits_month = st.number_input(
                "Mean Hits per Month", 
                min_value=0.0, 
                max_value=500.0, 
                value=4.0,
                help="Average hits per month (0-500, mean: 4.0)"
            )
            
            mean_pageviews_month = st.number_input(
                "Mean Pageviews per Month", 
                min_value=0.0, 
                max_value=466.0, 
                value=3.35,
                help="Average pageviews per month (0-466, mean: 3.35)"
            )
            
            month = st.selectbox(
                "Month", 
                list(range(1, 13)), 
                index=6,
                help="Month (1-12)"
            )
            
            year = st.selectbox(
                "Year", 
                [2016, 2017], 
                index=0,
                help="Year (2016-2017)"
            )

        # Create prediction button
        st.markdown("---")
        if st.button("🔮 Predict Revenue", type="primary", use_container_width=True):
            try:
                # Create feature vector
                features = create_feature_vector(
                    device_encoded, hits, mean_hits_month, 
                    mean_pageviews_month, month, pageviews, visit_number, year
                )
                
                # Make prediction
                prediction = model.predict(features, verbose=0)
                revenue = prediction[0][0]
                
                # Display result
                if revenue < 0:
                    revenue = 0
                
                # Create result display
                st.markdown("### 🎯 Prediction Results")
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    st.metric(
                        label="💰 Predicted Log Revenue", 
                        value=f"{revenue:.4f}",
                        help="Logarithmic scale prediction"
                    )
                
                with col_res2:
                    actual_revenue = np.exp(revenue)
                    st.metric(
                        label="💵 Estimated Revenue", 
                        value=f"${actual_revenue:.2f}",
                        help="Converted to actual revenue estimate"
                    )
                
                # Revenue interpretation
                if actual_revenue < 1:
                    st.info("📊 **Low Revenue**: Customer shows minimal monetization potential")
                elif actual_revenue < 10:
                    st.info("📊 **Medium Revenue**: Customer shows moderate monetization potential")
                else:
                    st.success("📊 **High Revenue**: Customer shows strong monetization potential")
                
                # Show feature vector for debugging
                with st.expander("🔍 Debug Info - Feature Vector"):
                    feature_names = [
                        'deviceCategory_encoded', 'hits', 'mean_hits_month',
                        'mean_pageviews_month', 'month', 'pageviews', 'visitNumber', 'year'
                    ]
                    debug_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Value': features.flatten(),
                        'Data Type': [type(x).__name__ for x in features.flatten()]
                    })
                    st.dataframe(debug_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
                st.write("Debug info - Feature vector shape:", features.shape if 'features' in locals() else "Not created")
    
    elif input_method == "Upload CSV":
        st.subheader("📁 Upload CSV File")
        
        uploaded_file = st.file_uploader(
            "Choose CSV file", 
            type="csv",
            help="Upload a CSV file with all 8 required features"
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                st.write("📊 **Data Preview:**")
                st.dataframe(df.head(), use_container_width=True)
                
                # Dataset info
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.metric("Rows", df.shape[0])
                with col_info2:
                    st.metric("Columns", df.shape[1])
                
                # Required columns for 8 features (CORRECT ORDER)
                required_cols = [
                    'deviceCategory_encoded', 'hits', 'mean_hits_month',
                    'mean_pageviews_month', 'month', 'pageviews', 'visitNumber', 'year'
                ]
                
                # Check for deviceCategory column (if not encoded yet)
                if 'deviceCategory' in df.columns and 'deviceCategory_encoded' not in df.columns:
                    device_mapping = {'desktop': 0, 'mobile': 1, 'tablet': 2}
                    df['deviceCategory_encoded'] = df['deviceCategory'].map(device_mapping)
                    st.info("✅ Device category automatically encoded")
                
                # Check columns
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.warning(f"⚠️ **Missing columns:** {', '.join(missing_cols)}")
                    
                    with st.expander("📋 Column Mapping Guide"):
                        st.write("**Required columns:**")
                        for i, col in enumerate(required_cols, 1):
                            status = "✅" if col in df.columns else "❌"
                            st.write(f"{i}. {status} {col}")
                        
                        st.write("**Available columns in your file:**")
                        for col in df.columns:
                            st.write(f"• {col}")
                else:
                    st.success(f"✅ All {len(required_cols)} required columns found!")
                
                if st.button("🔮 Predict All Revenue", type="primary", use_container_width=True) and not missing_cols:
                    with st.spinner("Making predictions..."):
                        # Select and reorder columns to match training data
                        feature_df = df[required_cols].copy()
                        
                        # Convert to float32
                        feature_df = feature_df.astype(np.float32)
                        
                        # Make predictions
                        predictions = model.predict(feature_df.values, verbose=0)
                        
                        # Add predictions to original dataframe
                        df['PredictedLogRevenue'] = predictions.flatten()
                        df['PredictedRevenue'] = np.exp(predictions.flatten())
                        
                        st.success("✅ Predictions completed!")
                        
                        # Show results
                        st.markdown("### 📊 Results")
                        st.dataframe(df, use_container_width=True)
                        
                        # Statistics
                        st.markdown("### 📈 Prediction Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Average Revenue", 
                                f"${df['PredictedRevenue'].mean():.2f}",
                                help="Mean predicted revenue"
                            )
                        with col2:
                            st.metric(
                                "Max Revenue", 
                                f"${df['PredictedRevenue'].max():.2f}",
                                help="Highest predicted revenue"
                            )
                        with col3:
                            st.metric(
                                "Total Predicted", 
                                f"${df['PredictedRevenue'].sum():.2f}",
                                help="Sum of all predicted revenues"
                            )
                        with col4:
                            st.metric(
                                "Records Processed", 
                                f"{len(df):,}",
                                help="Number of predictions made"
                            )
                        
                        # Revenue distribution
                        st.markdown("### 📊 Revenue Distribution")
                        
                        # Categorize predictions
                        low_revenue = (df['PredictedRevenue'] < 1).sum()
                        medium_revenue = ((df['PredictedRevenue'] >= 1) & (df['PredictedRevenue'] < 10)).sum()
                        high_revenue = (df['PredictedRevenue'] >= 10).sum()
                        
                        col_dist1, col_dist2, col_dist3 = st.columns(3)
                        
                        with col_dist1:
                            st.metric(
                                "Low Revenue (<$1)", 
                                f"{low_revenue} ({low_revenue/len(df)*100:.1f}%)",
                                help="Customers with minimal monetization potential"
                            )
                        with col_dist2:
                            st.metric(
                                "Medium Revenue ($1-$10)", 
                                f"{medium_revenue} ({medium_revenue/len(df)*100:.1f}%)",
                                help="Customers with moderate monetization potential"
                            )
                        with col_dist3:
                            st.metric(
                                "High Revenue (>$10)", 
                                f"{high_revenue} ({high_revenue/len(df)*100:.1f}%)",
                                help="Customers with strong monetization potential"
                            )
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv,
                            file_name="revenue_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
                st.write("Please check your CSV format and column names.")

else:
    st.error("❌ Could not load model. Please check model files.")
    st.markdown("""
    **Troubleshooting:**
    - Ensure `revenue_model.h5` is in the same directory
    - Check TensorFlow installation
    - Verify model file is not corrupted
    """)

# Enhanced sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Information")
st.sidebar.markdown("""
**Architecture:**
- Input Layer: 8 features
- Hidden Layers: Deep Neural Network
- Output Layer: 1 neuron (revenue)

**Performance:**
- Framework: TensorFlow/Keras
- Training: Train/Val/Test Split
- Optimization: Early Stopping + Hyperparameter Tuning
""")

# Feature statistics in sidebar
with st.sidebar.expander("📋 Feature Statistics"):
    st.markdown("""
    <b>Feature Ranges:</b><br><br>
    • deviceCategory_encoded: 0-2<br>
    • hits: 0-500 (mean: 4.0)<br>
    • mean_hits_month: 0-500 (mean: 4.0)<br>
    • mean_pageviews_month: 0-466 (mean: 3.35)<br>
    • month: 1-12<br>
    • pageviews: 0-469 (mean: 3.35)<br>
    • visitNumber: 0-395 (mean: 1.98)<br>
    • year: 2016-2017<br>
    """, unsafe_allow_html=True)

# Device encoding info
with st.sidebar.expander("📱 Device Encoding"):
    st.markdown("""
    **Device Categories:**
    - Desktop: 0
    - Mobile: 1  
    - Tablet: 2
    """)

# Footer
st.markdown("---")
st.markdown("""
### 💡 Usage Tips:
- **Manual Input**: Enter customer data to get individual revenue prediction
- **CSV Upload**: Upload CSV with 8 required features for batch predictions
- **Feature Values**: Use realistic values within the specified ranges
- **Revenue Output**: Model outputs log revenue, converted to actual $ estimates

### 📈 Model Features (8 total):
1. **deviceCategory_encoded**: Device type (0=Desktop, 1=Mobile, 2=Tablet)
2. **hits**: Number of hits per session
3. **mean_hits_month**: Average hits per month
4. **mean_pageviews_month**: Average pageviews per month
5. **month**: Month of visit (1-12)
6. **pageviews**: Number of pageviews per session
7. **visitNumber**: Visit sequence number
8. **year**: Year of visit (2016-2017)
""")

st.markdown("---")
st.markdown("**Built with Streamlit 🚀 | Powered by TensorFlow 🧠**")