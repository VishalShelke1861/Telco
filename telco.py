# ============================================================
# app.py
# TELCO CUSTOMER CHURN PREDICTION - STREAMLIT
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(

    page_title="Customer Churn Prediction",

    page_icon="📊",

    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(
        "customer_churn_pipeline.pkl"
    )


model = load_model()


# ============================================================
# TITLE
# ============================================================

st.title(
    "📊 Customer Churn Prediction"
)

st.write(
    "Predict whether a telecom customer is likely to churn."
)


st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Customer Information"
)


# ============================================================
# CUSTOMER INPUTS
# ============================================================

col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# COLUMN 1
# ------------------------------------------------------------

with col1:

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [
            "No",
            "Yes"
        ]
    )

    partner = st.selectbox(
        "Partner",
        [
            "No",
            "Yes"
        ]
    )

    dependents = st.selectbox(
        "Dependents",
        [
            "No",
            "Yes"
        ]
    )

    tenure = st.number_input(
        "Tenure Months",
        min_value=0,
        max_value=72,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        [
            "No",
            "Yes"
        ]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        [
            "No",
            "Yes",
            "No phone service"
        ]
    )


# ------------------------------------------------------------
# COLUMN 2
# ------------------------------------------------------------

with col2:

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    online_security = st.selectbox(
        "Online Security",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    online_backup = st.selectbox(
        "Online Backup",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    device_protection = st.selectbox(
        "Device Protection",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    tech_support = st.selectbox(
        "Tech Support",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )


# ------------------------------------------------------------
# COLUMN 3
# ------------------------------------------------------------

with col3:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        [
            "No",
            "Yes"
        ]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=1.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.0,
        step=10.0
    )

    cltv = st.number_input(
        "CLTV",
        min_value=0.0,
        value=4000.0,
        step=100.0
    )


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_churn(customer_data):

    customer_df = pd.DataFrame(
        [customer_data]
    )


    # --------------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------------

    # Feature 1: Tenure Group

    customer_df["Tenure_Group"] = pd.cut(

        customer_df["Tenure Months"],

        bins=[
            -1,
            6,
            12,
            24,
            48,
            72
        ],

        labels=[
            "0-6 Months",
            "7-12 Months",
            "13-24 Months",
            "25-48 Months",
            "49-72 Months"
        ]
    )


    # Feature 2: Average Monthly Spending

    customer_df[
        "Average_Monthly_Spending"
    ] = (

        customer_df[
            "Total Charges"
        ] /

        customer_df[
            "Tenure Months"
        ].replace(
            0,
            np.nan
        )
    )


    # Feature 3: CLTV

    customer_df[
        "Customer_Lifetime_Value"
    ] = customer_df[
        "CLTV"
    ]


    # Feature 4: Total Service Count

    service_columns = [

        "Phone Service",

        "Multiple Lines",

        "Online Security",

        "Online Backup",

        "Device Protection",

        "Tech Support",

        "Streaming TV",

        "Streaming Movies"

    ]


    customer_df[
        "Total_Service_Count"
    ] = (

        customer_df[
            service_columns
        ]

        .apply(

            lambda row:
            sum(
                value == "Yes"
                for value in row
            ),

            axis=1
        )
    )


    # Feature 5: Service Usage Indicator

    customer_df[
        "Service_Usage_Indicator"
    ] = (

        customer_df[
            "Total_Service_Count"
        ] >= 3

    ).astype(int)


    # Feature 6: Monthly Charge Category

    customer_df[
        "Monthly_Charge_Category"
    ] = pd.cut(

        customer_df[
            "Monthly Charges"
        ],

        bins=[
            -np.inf,
            40,
            70,
            np.inf
        ],

        labels=[
            "Low",
            "Medium",
            "High"
        ]
    )


    # --------------------------------------------------------
    # Remove columns not used during training
    # --------------------------------------------------------

    remove_columns = [

        "CustomerID",
        "Count",
        "Country",
        "State",
        "City",
        "Zip Code",
        "Lat Long",
        "Latitude",
        "Longitude",

        "Churn Value",
        "Churn Score",
        "Churn Reason"

    ]


    customer_df = customer_df.drop(

        columns=remove_columns,

        errors="ignore"

    )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(
        customer_df
    )[0]


    probability = model.predict_proba(
        customer_df
    )[0, 1]


    return prediction, probability


# ============================================================
# PREDICT BUTTON
# ============================================================

st.divider()


if st.button(
    "🔮 Predict Churn",
    type="primary",
    use_container_width=True
):

    # Create customer dictionary

    customer_data = {

        "Gender":
            gender,

        "Senior Citizen":
            senior_citizen,

        "Partner":
            partner,

        "Dependents":
            dependents,

        "Tenure Months":
            tenure,

        "Phone Service":
            phone_service,

        "Multiple Lines":
            multiple_lines,

        "Internet Service":
            internet_service,

        "Online Security":
            online_security,

        "Online Backup":
            online_backup,

        "Device Protection":
            device_protection,

        "Tech Support":
            tech_support,

        "Streaming TV":
            streaming_tv,

        "Streaming Movies":
            streaming_movies,

        "Contract":
            contract,

        "Paperless Billing":
            paperless_billing,

        "Payment Method":
            payment_method,

        "Monthly Charges":
            monthly_charges,

        "Total Charges":
            total_charges,

        "CLTV":
            cltv

    }


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    try:

        prediction, probability = (
            predict_churn(
                customer_data
            )
        )


        st.subheader(
            "Prediction Result"
        )


        # ----------------------------------------------------
        # CHURN
        # ----------------------------------------------------

        if prediction == 1:

            st.error(
                "⚠️ Customer is likely to churn"
            )

        else:

            st.success(
                "✅ Customer is unlikely to churn"
            )


        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        st.metric(

            "Churn Probability",

            f"{probability:.2%}"

        )


        # ----------------------------------------------------
        # PROGRESS BAR
        # ----------------------------------------------------

        st.write(
            "Churn Probability"
        )

        st.progress(
            float(probability)
        )


        # ----------------------------------------------------
        # RISK LEVEL
        # ----------------------------------------------------

        if probability >= 0.70:

            risk = "High Risk"

            st.error(
                f"🔴 {risk}"
            )

        elif probability >= 0.40:

            risk = "Medium Risk"

            st.warning(
                f"🟠 {risk}"
            )

        else:

            risk = "Low Risk"

            st.success(
                f"🟢 {risk}"
            )


        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        st.subheader(
            "Customer Summary"
        )


        summary = pd.DataFrame({

            "Parameter": [

                "Tenure Months",

                "Contract",

                "Internet Service",

                "Monthly Charges",

                "Total Charges",

                "Churn Probability",

                "Risk Level"

            ],

            "Value": [

                tenure,

                contract,

                internet_service,

                f"₹{monthly_charges:.2f}",

                f"₹{total_charges:.2f}",

                f"{probability:.2%}",

                risk

            ]

        })


        st.table(
            summary
        )


    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )