import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load Dataset
st.title("Machine Learning with Categorical Data")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())

    # Select Target Column
    target_column = st.selectbox("Select Target Column", df.columns)
    features = st.multiselect("Select Feature Columns", [col for col in df.columns if col != target_column])

    if features:
        X = df[features]
        y = df[target_column]

        # Identify Categorical Columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist() # convert column names into python list
        
        if categorical_cols:
            # Apply Label Encoding (simpler approach)
            label_encoders = {}
            for col in categorical_cols:
                encoder = LabelEncoder()
                X[col] = encoder.fit_transform(X[col])
                label_encoders[col] = encoder

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        st.write(f"Model Accuracy: {accuracy:.2f}")

        # User Input for Predictions
        input_data = {feature: st.text_input(f"Enter {feature}") if feature in categorical_cols else st.number_input(f"Enter {feature}") for feature in features}
        
        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])

            # Convert categorical values to encoded format
            for col in categorical_cols:
                input_df[col] = label_encoders[col].transform([input_df[col][0]])  # Transform single value

            prediction = model.predict(input_df)
            st.write(f"Predicted Class: {prediction[0]}")