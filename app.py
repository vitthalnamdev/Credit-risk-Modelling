import streamlit as st
import pickle 
import joblib
import warnings

warnings.filterwarnings("ignore", message="missing ScriptRunContext! This warning can be ignored when running in bare mode.")

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency, f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings
warnings.filterwarnings('ignore')

# sklearn.set_config(transform_output='pandas')

df = pd.read_csv('cleaned_data.csv')
df1 = df.copy()

st.set_page_config(
	page_title='Credit Risk Modeling',
	page_icon='🏦',
	layout='wide'
)

st.title('Reducing Credit Risk')

# INPUT
pct_tl_open_L6M = st.number_input('Percent accounts opened in last 6 months: ', min_value=0.0, max_value=100.0, step=0.1)
pct_tl_closed_L6M = st.number_input('Percent accounts closed in last 6 months: ', min_value=0.0, max_value=100.0, step=0.1)
Total_TL_closed_L12M = st.number_input('Total accounts closed in last 12 months: ', min_value=0, step=1)
pct_tl_closed_L12M = st.number_input('Percent accounts opened in last 12 months: ', min_value=0.0, max_value=100.0, step=0.1)
Tot_Missed_Pmnt = st.number_input('Total missed Payments: ', min_value=0, step=1)
CC_TL = st.number_input('Count of Credit card accounts: ', min_value=0, step=1)
Home_TL = st.number_input('Count of Housing Loan accounts: ', min_value=0, step=1)
PL_TL = st.number_input('Count of Personal Loan accounts: ', min_value=0, step=1)
Secured_TL=st.number_input('Count of secured accounts: ', min_value=0, step=1)
Unsecured_TL = st.number_input('Count of unsecured accounts: ', min_value=0, step=1)
Other_TL = st.number_input('Count of other accounts: ', min_value=0, step=1)
Age_Oldest_TL = st.number_input('Age of oldest opened account(in days): ', min_value=0, step=1)
Age_Newest_TL = st.number_input('Age of newest opened account(in days): ', min_value=0, step=1)
time_since_recent_payment = st.number_input('Time Since recent Payment made(in days): ', min_value=0, step=1)
max_recent_level_of_deliq = st.number_input('Maximum recent level of delinquency(in days): ', min_value=0, step=1)
num_deliq_6_12mts = st.number_input('Number of times delinquent between last 6 months and last 12 months: ', min_value=0, step=1)
num_times_60p_dpd = st.number_input('Number of times 60+ dpd: ', min_value=0, step=1)
num_std_12mts = st.number_input('Number of standard Payments in last 12 months: ', min_value=0, step=1)
num_sub = st.number_input('Number of sub standard payments - not making full payments: ', min_value=0, step=1)
num_sub_6mts = st.number_input('Number of sub standard payments in last 6 months: ', min_value=0, step=1)
num_sub_12mts = st.number_input('Number of sub standard payments in last 12 months: ', min_value=0, step=1)
num_dbt = st.number_input('Number of doubtful payments: ', min_value=0, step=1)
num_dbt_12mts = st.number_input('Number of doubtful payments in last 12 months: ', min_value=0, step=1)
num_lss = st.number_input('Number of loss accounts: ', min_value=0, step=1)
recent_level_of_deliq = st.number_input('Recent level of delinquency: ', min_value=0, step=1)
CC_enq_L12m = st.number_input('Credit card enquiries in last 12 months: ', min_value=0, step=1)
PL_enq_L12m = st.number_input('Personal loan enquiries in last 12 months: ', min_value=0, step=1)
time_since_recent_enq = st.number_input('Time since recent enquiry(in days): ', min_value=0, step=1)
enq_L3m = st.number_input('Enquiries in last 3 months: ', min_value=0, step=1)
NETMONTHLYINCOME = st.number_input('net monthly income: ', min_value=0, step=1)
Time_With_Curr_Empr = st.number_input('Time with current Employer: ', min_value=0, step=1)
CC_Flag = st.number_input('is credit card taken: ', min_value=0,max_value=1, step=1)
PL_Flag = st.number_input('Is personal loan taken: ', min_value=0,max_value=1, step=1)
pct_PL_enq_L6m_of_ever = st.number_input('Percent enquiries for personal loan in last 6 months to last 6 months: ', min_value=0.0, step=0.1)
pct_CC_enq_L6m_of_ever = st.number_input('Percent enquiries for Credit card in last 6 months to last 6 months: ', min_value=0.0, step=0.1)
HL_Flag = st.number_input('Is housing Loan taken: ', min_value=0,max_value=1, step=1)
GL_Flag = st.number_input('Is Goald Loan taken: ', min_value=0, max_value=1, step=1)
MARITALSTATUS = st.selectbox('Marital Status: ', options=df.MARITALSTATUS.unique())
EDUCATION = st.selectbox('Education level: ', options=df.EDUCATION.unique())
GENDER = st.selectbox('Gender: ', options=df.GENDER.unique())
first_prod_enq2 = st.selectbox('First product enquired For : ', options=df.last_prod_enq2.unique())
last_prod_enq2 = st.selectbox('Last product enquired For : ', options=df.last_prod_enq2.unique())

X_new = pd.DataFrame(dict(
	pct_tl_open_L6M = [pct_tl_open_L6M],
	pct_tl_closed_L6M = [pct_tl_closed_L6M],
	Total_TL_closed_L12M = [Total_TL_closed_L12M],
	pct_tl_closed_L12M = [pct_tl_closed_L12M],
	Tot_Missed_Pmnt = [Tot_Missed_Pmnt],
	CC_TL = [CC_TL],
	Home_TL = [Home_TL],
	PL_TL = [PL_TL],
	Secured_TL = [Secured_TL],
	Unsecured_TL = [Unsecured_TL],
	Other_TL = [Other_TL],
	Age_Oldest_TL = [Age_Oldest_TL],
	Age_Newest_TL = [Age_Newest_TL],
	time_since_recent_payment = [time_since_recent_payment],
	max_recent_level_of_deliq = [max_recent_level_of_deliq],
	num_deliq_6_12mts = [num_deliq_6_12mts],
	num_times_60p_dpd = [num_times_60p_dpd],
	num_std_12mts = [num_std_12mts],
	num_sub = [num_sub],
	num_sub_6mts = [num_sub_6mts],
	num_sub_12mts = [num_sub_12mts],
	num_dbt = [num_dbt],
	num_dbt_12mts = [num_dbt_12mts],
	num_lss = [num_lss],
	recent_level_of_deliq = [recent_level_of_deliq],
	CC_enq_L12m = [CC_enq_L12m],
	PL_enq_L12m = [PL_enq_L12m],
	time_since_recent_enq = [time_since_recent_enq],
	enq_L3m = [enq_L3m],
	NETMONTHLYINCOME = [NETMONTHLYINCOME],
	Time_With_Curr_Empr = [Time_With_Curr_Empr],
	CC_Flag = [CC_Flag],
	PL_Flag = [PL_Flag],
	pct_PL_enq_L6m_of_ever = [pct_PL_enq_L6m_of_ever],
	pct_CC_enq_L6m_of_ever = [pct_CC_enq_L6m_of_ever],
	HL_Flag = [HL_Flag],
	GL_Flag = [GL_Flag],
	MARITALSTATUS = [MARITALSTATUS],
	EDUCATION = [EDUCATION],
	GENDER = [GENDER],
	first_prod_enq2 = [first_prod_enq2],
	last_prod_enq2 = [last_prod_enq2],
))

X_new.loc[X_new['EDUCATION']=='SSC', ['EDUCATION']] = 1
X_new.loc[X_new['EDUCATION']=='12TH', ['EDUCATION']] = 2
X_new.loc[X_new['EDUCATION']=='GRADUATE', ['EDUCATION']] = 3
X_new.loc[X_new['EDUCATION']=='UNDER GRADUATE', ['EDUCATION']] = 3
X_new.loc[X_new['EDUCATION']=='POST-GRADUATE', ['EDUCATION']] = 4
X_new.loc[X_new['EDUCATION']=='OTHERS', ['EDUCATION']] = 1
X_new.loc[X_new['EDUCATION']=='PROFESSIONAL',['EDUCATION']] = 3  


X_new['EDUCATION'] = X_new['EDUCATION'].astype(int)
X_new_encoded = pd.get_dummies(X_new, columns=['MARITALSTATUS','GENDER','last_prod_enq2','first_prod_enq2'], dtype='int')




xgb_model = pickle.load(open('xgboost_model.pkl', 'rb'))

model_features = xgb_model.get_booster().feature_names
X_new_encoded = X_new_encoded.reindex(columns=model_features, fill_value=0)

if st.button('Predict'):
	result = xgb_model.predict(X_new_encoded)[0]

	if result == 0:
		st.info('P1')
	elif result == 1:
		st.info('P2')
	elif result == 2:
		st.info('P3')
	elif result == 3:
		st.info('P4')




