#%%
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator

import pickle
import base64

class savgol_transformer(BaseEstimator):
    def __init__(self, window_length, polyorder, deriv):
        self.winlen = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        
    def fit(self, data, y=None):
        return self
    
    def transform(self, x_dataset):
        X_trans = savgol_filter(x_dataset, window_length=self.winlen, polyorder=self.polyorder, deriv=self.deriv)
        
        return X_trans
def download_link(object_to_download, download_filename, download_link_text):
    """
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode("UTF-8")).decode()#encode as csv

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'
#%%
trained_model = st.file_uploader("Upload your trained PLS Model here")
if trained_model is not None:
    loaded_model = pickle.load(trained_model)

    st.write("Model Loaded")
else:
    st.exception(exception=NameError("Import a trained model in order to initiate the app"))
    st.stop()
    
#%%
st.title("Partial Least Squares Cis-DP Online Prediction by IR")

    
### Load in the test data ###
uploaded_csv = st.file_uploader("Upload your spectra for prediction")
if uploaded_csv is not None:
    test_data = pd.read_csv(uploaded_csv, header=None)

    #get X test matrix
    X_test = test_data.values[:, :]
    y_pred = loaded_model.predict(X_test)
    y_pred_df = pd.DataFrame({"predicted_mg/mL":(y_pred.reshape(1, -1)).flatten()})
    y_pred_chart = alt.Chart(y_pred_df).mark_bar().encode(
        alt.X("predicted_mg/mL", bin=True),
        y='count()',
    )
    st.altair_chart(y_pred_chart, use_container_width=True)
    st.write("Predicted Mean Concentration = {:.2f} mg/mL   \nStandard Deviation = {:.2f} mg/mL".format(np.mean(y_pred), np.std(y_pred)))
    #st.write("Expected Concentration by HPLC = 9.6 mg/mL")
    df_out = pd.DataFrame({"Predicted mg/mL":(y_pred.flatten())})
    st.table(df_out)
    #download predicted result table
    if st.button('Download Results as CSV'):
        tmp_download_link = download_link(df_out, 'Predicted_Results.csv', 'Click here to download your data!')
    st.markdown(tmp_download_link, unsafe_allow_html=True)
