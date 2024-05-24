import streamlit as st
import pickle
import numpy as np
rf = pickle.load(open('rf_model_of_IRIS_proj.pkl','rb'))

st.title('Iris Flower Prediction Web App')
st.subheader('')

sepal_length = st.slider('sepal_length', 4.3, 7.9, 5.4)
sepal_width = st.slider('sepal_width', 2.0, 4.4, 3.4)
petal_length = st.slider('petal_length', 1.0, 6.9, 1.3)
petal_width = st.slider('petal_width', 0.1, 2.5, 0.2)
test = np.array([sepal_length,sepal_width,petal_length,petal_width])
test = test.reshape(1,4)
if st.button('Predict'):
    st.success(rf.predict(test)[0])



# To run Streamlit Web App
# streamlit run app.py

