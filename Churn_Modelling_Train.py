import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


st.title('Churn Classifier')
st.write(
    """This app uses 10 inputs to predict likelihood of churn!"""
)

with st.form('input'):
    churn_file = st.file_uploader('Upload your churn data')
    st.form_submit_button()

    if churn_file is None:
        rf_pickle = open("random_forest_churn.pickle", "rb")
        rfc = pickle.load(rf_pickle)
        std_pickle = open("std.pickle", "rb")
        scaler = pickle.load(std_pickle)
        rf_pickle.close()
        std_pickle.close()
    
    else:
        df = pd.read_csv(churn_file)
        # Shuffle the data
        #np.random.seed(42) 
        df = df.reindex(np.random.permutation(df.index))
        df.reset_index(inplace=True, drop=True) # Reset index
        
        # Remove 'RowNumber','CustomerId','Surname' features
        df = df.drop(['RowNumber','CustomerId','Surname'],axis=1,inplace=False)
        
        # Training and Test
        spt = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_idx, test_idx in spt.split(df, df['Exited']):
            train_set_strat = df.loc[train_idx]
            test_set_strat  = df.loc[test_idx] 
            
        train_set_strat.reset_index(inplace=True, drop=True) # Reset index
        test_set_strat.reset_index(inplace=True, drop=True) # Reset index  
    
        # Text Handeling
        # Convert Geography to one-hot-encoding
        Geog_1hot=pd.get_dummies(train_set_strat['Geography'],prefix='Is')
        
        # Convert gender to 0 and 1
        ordinal_encoder = OrdinalEncoder()
        train_set_strat['Gender'] = ordinal_encoder.fit_transform(train_set_strat[['Gender']])
        
        # Remove 'Geography'
        train_set_strat = train_set_strat.drop(['Geography'],axis=1,inplace=False)
        train_set_strat = pd.concat([Geog_1hot,train_set_strat], axis=1) # Concatenate rows
    
    
        # Standardization 
        # Make training features and target
        X_train = train_set_strat.drop("Exited", axis=1)
        y_train = train_set_strat["Exited"].values
        
        # Divide into two training sets (with and without standization)
        clmn=['Is_France','Is_Germany','Is_Spain',
              'Gender','NumOfProducts','HasCrCard','IsActiveMember']
        X_train_for_std = X_train.drop(clmn, axis=1)
        X_train_not_std = X_train[clmn]
        features_colums=list(X_train_for_std.columns)+list(X_train_not_std.columns)
        #
        scaler = StandardScaler()
        scaler.fit(X_train_for_std)
        #
        df_train_std = scaler.transform(X_train_for_std)
        X_train_std = np.concatenate((df_train_std,X_train_not_std), axis=1)
    
    
        # Random Forest for taining set
        rnd = RandomForestClassifier(n_estimators=50, max_depth= 25, min_samples_split= 20, bootstrap= True, random_state=42)
        rnd.fit(X_train_std,y_train)
        
        ################### Data Processing for test set  ######################
        # Convert Geography to one-hot-encoding
        Geog_1hot = pd.get_dummies(test_set_strat['Geography'],prefix='Is')
        
        # Convert gender to 0 and 1
        ordinal_encoder = OrdinalEncoder()
        test_set_strat['Gender'] = ordinal_encoder.fit_transform(test_set_strat[['Gender']])
        
        # Remove 'Geography'
        test_set_strat = test_set_strat.drop(['Geography'],axis=1,inplace=False)
        test_set_strat = pd.concat([Geog_1hot,test_set_strat], axis=1) # Concatenate rows
        
        # Standardize data
        X_test = test_set_strat.drop("Exited", axis=1)
        y_test = test_set_strat["Exited"].values
        #
        clmn = ['Is_France','Is_Germany','Is_Spain',
              'Gender','NumOfProducts','HasCrCard','IsActiveMember']
        X_test_for_std = X_test.drop(clmn, axis=1)
        X_test_not_std = X_test[clmn]
        features_colums=list(X_test_for_std.columns)+list(X_test_not_std.columns)
        #
        df_test_std = scaler.transform(X_test_for_std)
        X_test_std = np.concatenate((df_test_std,X_test_not_std), axis=1)
        
        # Random Forest for test set
        y_test_pred = rnd.predict(X_test_std)
        y_test_proba_rnd = rnd.predict_proba(X_test_std)
        
        score = accuracy_score(y_test_pred, y_test)
        print(f'The accuracy score for this model is {score}')
    
    
    
        st.write(
            f"""We trained a Random Forest model on these
            data. The accuracy score for this model is {score}."""
        )


with st.form('user_inputs'):
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

    st.form_submit_button()
    
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
    
    
    clmn_std = np.array([CreditScore, Age, Tenure, Balance, EstimatedSalary]).reshape(1, 5)
    clmn_not_std = np.array([Is_France, Is_Germany, Is_Spain,
          Gender, NumOfProducts, HasCrCard, IsActiveMember]).reshape(1, 7)
    feat_std = scaler.transform(clmn_std)
    to_pred = np.concatenate((feat_std, clmn_not_std), axis=1)
    
    
    y_pred = int(rnd.predict_proba(to_pred)[0][0]*100)
    st.write(f"Predict churn for this customer is {y_pred}%")


st.subheader("Feature Importance by Loading Image")
st.write(f"We predict churn using customers input")
st.write(
    """We used a machine learning (Random Forest)
    model to predict Customer Churn."""
)

st.write(
    """Here is the Figure loaded as image in Streamlit."""
)    
st.image('download.png')


st.subheader("Feature Importance by Generating Image")

import pandas as pd
import matplotlib
import pylab as plt
from matplotlib.ticker import PercentFormatter

class prfrmnce_plot(object):
    """Plot performance of features to predict a target"""
    def __init__(self,importance: list, title: str, ylabel: str,clmns: str,
                titlefontsize: int=10, xfontsize: int=5, yfontsize: int=8) -> None:
        self.importance    = importance
        self.title         = title 
        self.ylabel        = ylabel  
        self.clmns         = clmns  
        self.titlefontsize = titlefontsize 
        self.xfontsize     = xfontsize 
        self.yfontsize     = yfontsize
        
    #########################    
    
    def bargraph(self, select: bool= False, fontsizelable: bool= False, xshift: float=-0.1, nsim: int=False
                 ,yshift: float=0.01,perent: bool=False, xlim: list=False,axt=None,
                 ylim: list=False, y_rot: int=0, graph_float: bool=True) -> pd.DataFrame():
        ax1 = axt or plt.axes()
        if not nsim:
            # Make all negative coefficients to positive
            sort_score=sorted(zip(abs(self.importance),self.clmns), reverse=True)
            Clmns_sort=[sort_score[i][1] for i in range(len(self.clmns))]
            sort_score=[sort_score[i][0] for i in range(len(self.clmns))]
        else:
            importance_agg=[]
            importance_std=[]
            for iclmn in range(len(self.clmns)):
                tmp=[]
                for isim in range(nsim):
                    tmp.append(abs(self.importance[isim][iclmn]))
                importance_agg.append(np.mean(tmp))
                importance_std.append(np.std(tmp))
                
            # Make all negative coefficients to positive
            sort_score=sorted(zip(importance_agg,self.clmns), reverse=True)
            Clmns_sort=[sort_score[i][1] for i in range(len(self.clmns))]
            sort_score=[sort_score[i][0] for i in range(len(self.clmns))]                
            

        index1 = np.arange(len(self.clmns))
        # select the most important features
        if (select):
            Clmns_sort=Clmns_sort[:select]
            sort_score=sort_score[:select]
        ax1.bar(Clmns_sort, sort_score, width=0.6, align='center', alpha=1, edgecolor='k', capsize=4,color='b')
        plt.title(self.title,fontsize=self.titlefontsize)
        ax1.set_ylabel(self.ylabel,fontsize=self.yfontsize)
        ax1.set_xticks(np.arange(len(Clmns_sort)))
        
        ax1.set_xticklabels(Clmns_sort,fontsize=self.xfontsize, rotation=90,y=0.02)   
        if (perent): plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        ax1.xaxis.grid(color='k', linestyle='--', linewidth=0.2) 
        if (xlim): plt.xlim(xlim)
        if (ylim): plt.ylim(ylim)
        if (fontsizelable):
            for ii in range(len(sort_score)):
                if (perent):
                    plt.text(xshift+ii, sort_score[ii]+yshift,f'{"{0:.1f}".format(sort_score[ii]*100)}%',
                    fontsize=fontsizelable,rotation=y_rot,color='k')     
                else:
                    if graph_float:
                        plt.text(xshift+ii, sort_score[ii]+yshift,f'{"{0:.3f}".format(sort_score[ii])}',
                        fontsize=fontsizelable,rotation=y_rot,color='k') 
                    else:
                        plt.text(xshift+ii, sort_score[ii]+yshift,f'{"{0:.0f}".format(sort_score[ii])}',
                            fontsize=fontsizelable,rotation=y_rot,color='k')                             
                    
        
        dic_Clmns={}
        for i in range(len(Clmns_sort)):
            dic_Clmns[Clmns_sort[i]]=sort_score[i]
            

 # Plot the importance of features
font = {'size'   : 7}
plt.rc('font', **font)
fig, ax1 = plt.subplots(figsize=(6, 3), dpi= 180, facecolor='w', edgecolor='k')


# Calculate importance
importance = abs(rnd.feature_importances_)

df_most_important = prfrmnce_plot(importance, title=f'Feature Importance by Random Forest', 
            ylabel='Random Forest Score',clmns=features_colums,titlefontsize=9, 
            xfontsize=7, yfontsize=8).bargraph(perent=True,fontsizelable=8,xshift=-0.25,axt=ax1,
            yshift=0.01,ylim=[0,0.4], xlim=[-0.5,11.5], y_rot=0)      
            
st.pyplot(fig)             