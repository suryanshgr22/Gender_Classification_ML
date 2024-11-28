import numpy as np
import pickle 
import streamlit as st

loaded_model = pickle.load(open('https://github.com/suryanshgr22/Gender_Classification_ML/blob/main/trained_model.sav', 'rb'))

def prediction(input):
    # input = [0,	14.0,	5.4,	0,	0,	1,	0]
    input_as_numpy_array = np.asarray(input)
    input_reshaped = input_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_reshaped)
    if(prediction[0] == 1):
        return "MALE"
    else:
        return "FEMALE"
    
def main():

    st.title("Gender Prediction")
    #forehead_height_cm,nose_wide,nose_long,lips_thin,distance_nose_to_lip_long,

    long_hair = st.text_input("Long Hair")
    forehead_width_cm = st.text_input("forehead_width_cm")
    forehead_height_cm = st.text_input("forehead_height_cm")
    nose_wide = st.text_input("nose_wide")
    nose_long = st.text_input("nose_long")
    lips_thin = st.text_input("lips_thin")
    distance_nose_to_lip_long = st.text_input("distance_nose_to_lip_long")

    gender_outcome = ''

    if(st.button("Predict Gender")):
        gender_outcome = prediction([long_hair,forehead_width_cm,forehead_height_cm,nose_wide,nose_long,lips_thin,distance_nose_to_lip_long])

    st.success(gender_outcome)


if __name__ == 'main':
    main()


