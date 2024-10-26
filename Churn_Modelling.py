import pickle
import streamlit as st
import numpy as np

st.title('Customer Churn Prediction')

st.write(
    """This app is created by [streamlit](https://streamlit.io/) to predict the likelihood if bank 
    customers will turnover next cycle. Random forest classifier is trained by Bank Turnover Dataset from 
    [Kaggle](https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling/version/1).
     The app is trained by 10 inputs (predictors). The user inputs are `Geography`, `CreditScore`, 
     `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`"""
)
st.image('DataTable.jpg')

rf_pickle = open("random_forest_churn.pickle", "rb")
rfc = pickle.load(rf_pickle)

rf_pickle.close()

st.write('')

col1, col2, col3  = st.columns(3)

col1.subheader("Input Data")
Geography = col1.selectbox("Geography", options=["France", "Germany", "Spain"])
CreditScore = col1.number_input("CreditScore", min_value=300)
Gender = col1.selectbox("Gender", options=['Male', 'Female'])
Age = col1.number_input("Age", min_value=18)
Tenure = col1.number_input("Tenure", min_value=2)

col2.subheader(" ")
col2.subheader(" ")
Balance = col2.number_input("Balance", min_value=500)
NumOfProducts = col2.number_input("NumOfProducts", min_value=1)
HasCrCard = col2.selectbox("HasCrCard", options=[0, 1])
IsActiveMember = col2.selectbox("IsActiveMember", options=[0, 1])
EstimatedSalary = col2.number_input("EstimatedSalary", min_value=1000)

user_inputs = [Geography, CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]


Is_France, Is_Germany, Is_Spain = 0, 0, 0
if Geography == 'France':
    Is_France = 1
elif Geography == 'Germany':
    Is_Germany = 1
elif Geography == 'Spain':
    Is_Spain = 1

if Gender == 'Male':
    Gender = 1
elif Gender == 'Female':
    Gender = 0


std_pickle = open("std.pickle", "rb")
scaler = pickle.load(std_pickle)
std_pickle.close()


clmn_std = np.array([CreditScore, Age, Tenure, Balance, EstimatedSalary]).reshape(1, 5)
clmn_not_std = np.array([Is_France, Is_Germany, Is_Spain,
      Gender, NumOfProducts, HasCrCard, IsActiveMember]).reshape(1, 7)
feat_std = scaler.transform(clmn_std)
to_pred = np.concatenate((feat_std, clmn_not_std), axis=1)


y_pred = int(rfc.predict_proba(to_pred)[0][0]*100)

col3.subheader("Prediction")
col3.write(f"The likelihood of churn for this customer is predicted **{y_pred}**%")
