import pickle
import streamlit as st
import numpy as np


rf_pickle = open("random_forest_churn.pickle", "rb")
rfc = pickle.load(rf_pickle)

rf_pickle.close()

Geography = st.selectbox("Geography", options=["France", "Germany", "Spain"])
CreditScore = st.number_input("CreditScore", min_value=300)
Gender = st.selectbox("Gender", options=['Male', 'Female'])
Age = st.number_input("Age", min_value=18)
Tenure = st.number_input("Tenure", min_value=2)
Balance = st.number_input("Balance", min_value=500)
NumOfProducts = st.number_input("NumOfProducts", min_value=1)
HasCrCard = st.selectbox("HasCrCard", options=[0, 1])
IsActiveMember = st.selectbox("IsActiveMember", options=[0, 1])
EstimatedSalary = st.number_input("EstimatedSalary", min_value=1000)

user_inputs = [Geography, CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]
st.write("The user inputs are `Geography`, `CreditScore`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`")

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


st.write(f"Predict churn for this customer is **{y_pred}**%")
