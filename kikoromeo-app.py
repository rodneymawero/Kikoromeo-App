import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pickled model
with open('catboost_clf.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# load the label binarizer 
with open('lbl.pkl', 'rb') as f:
    lbl = pickle.load(f)

# Save feature names
#with open('feature_names.pkl', 'wb') as f:
#    pickle.dump(feature_names.columns.tolist(), f)


#features_list = (['age', 'sex', 'on thyroxine', 'query on thyroxine',
#      'on antithyroid medication', 'sick', 'pregnant', 'thyroid surgery',
#      'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium',
#      'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured', 'TSH',
#       'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured', 'T4U',
#       'FTI measured', 'FTI', 'referral source'])

# Create the Streamlit app
def main():
    st.title("Kikoromeo-App - Hypothyroid Disease Prediction App" )

    st.write("Welcome to the Disease Prediction App created by Mawero Rodney G. 😎😎. He is a Kenyan. \U0001F1F0\U0001F1EA")

    st.write("**Kikoromeo** is a word in Kiswahili that means **thyroid gland**")

    st.write("The dataset used to build the model is the Thyroid Dataset (Garavan Institute) from the UCI Machine Learning Repository  https://archive.ics.uci.edu/dataset/102/thyroid+disease. ")

    st.write("Courtesy Ross Quinlan")

    # Display animated GIF
    st.image("kikoromeo-app.gif", caption="kikoromeo-app", use_column_width=True)
    st.title("Inspirational Quotes Flashcards")

    quotes = [
        ("Παν μέτρον άριστον - Pan Metron Ariston- Everything in Moderation" , "Kleobulus of Lindos -One of the Seven Sages of Ancient Greece."),
        ("Believe you can and you're halfway there.", "Theodore Roosevelt"),
        ("The only way to do great work is to love what you do.", "Steve Jobs"),
        ("In the middle of every difficulty lies opportunity.", "Albert Einstein"),
        ("Success is not final, failure is not fatal: It is the courage to continue that counts.", "Winston Churchill"),
        ("The future belongs to those who believe in the beauty of their dreams.", "Eleanor Roosevelt")]

    for i, (quote, author) in enumerate(quotes, start=1):
        with st.expander(f"Quote {i}"):
            st.write(f"\"{quote}\"")
            st.write(f"- {author}")



    # Create radio buttons for categorical columns snd sliders for numerical columns
    # Sidebar
    st.sidebar.title("App Settings")

    age = st.sidebar.slider('age', min_value=0, max_value=100)
    sex = st.sidebar.radio('sex', ('Female', 'Male'))
    on_thyroxine = st.sidebar.radio('on thyroxine', ('Yes', 'No'))
    query_on_thyroxine = st.sidebar.radio('query on thyroxine', ('Yes', 'No'))
    on_antithyroid_medication = st.sidebar.radio('on antithyroid medication', ('Yes', 'No'))
    sick = st.sidebar.radio('sick', ('Yes', 'No'))
    pregnant = st.sidebar.radio('pregnant', ('Yes', 'No'))
    thyroid_surgery = st.sidebar.radio('thyroid surgery', ('Yes', 'No'))
    I131_treatment = st.sidebar.radio('I131 treatment', ('Yes', 'No'))
    query_hypothyroid = st.sidebar.radio('query hypothyroid', ('Yes', 'No'))
    query_hyperthyroid = st.sidebar.radio('query hyperthyroid', ('Yes', 'No'))
    lithium = st.sidebar.radio('lithium', ('Yes', 'No'))
    goitre = st.sidebar.radio('goitre', ('Yes', 'No'))
    tumor = st.sidebar.radio('tumor', ('Yes', 'No'))
    hypopituitary = st.sidebar.radio('hypopituitary', ('Yes', 'No'))
    psych = st.sidebar.radio('psych', ('Yes', 'No'))
    TSH_measured = st.sidebar.radio('TSH_measured', ('Yes', 'No'))
    TSH = st.sidebar.slider('TSH: Thyroid Stimulating Hormone', min_value=0.0, max_value=200.0)
    T3_measured = st.sidebar.radio('18', ('Yes', 'No'))
    T3 = st.sidebar.slider('T3', min_value=0.0, max_value=10.0)
    TT4_measured = st.sidebar.radio('20', ('Yes', 'No'))
    TT4 = st.sidebar.slider('TT4', min_value=0.0, max_value=400.0)
    T4U_measured = st.sidebar.radio('T4U_measured', ('Yes', 'No'))
    T4U = st.sidebar.slider('T4U', min_value=0.0, max_value=3.0)
    FTI_measured = st.sidebar.radio('25', ('Yes', 'No'))
    FTI = st.sidebar.slider('FTI: Free Thyroxine Index', min_value=0.0, max_value=300.0)
    referral_source = st.sidebar.radio('referral source', (5, 4, 3, 2, 1))


    # Main content
    st.title("User inputs and disease prediction")
    st.write("Instructions: Please make your selections on the **sidebar** on the left, then click predict to get your prediction")

    st.write(f"Age: {age}")
    st.write(f"Sex: {sex}")
    st.write(f"On Thyroxine: {on_thyroxine}")
    st.write(f"Query on thyroxine: {query_on_thyroxine}")
    st.write(f"On antithyroid medication: {on_antithyroid_medication}")
    st.write(f"Sick: {sick}")
    st.write(f"Pregnant: {pregnant}")
    st.write(f"Thyroid surgery: {thyroid_surgery}")
    st.write(f"I131 treatment: {I131_treatment}")
    st.write(f"Query hypothyroid: {query_hypothyroid}")
    st.write(f"Query hyperthyroid: {query_hyperthyroid}")
    st.write(f"Lithium: {lithium}")
    st.write(f"Goitre: {goitre}")
    st.write(f"Tumor: {tumor}")
    st.write(f"Hypopituitary: {hypopituitary}")
    st.write(f"Psych: {psych}")
    st.write(f"TSH measured: {TSH_measured}")
    st.write(f"TSH: {TSH}")
    st.write(f"T3 measured: {T3_measured}")
    st.write(f"T3: {T3}")
    st.write(f"TT4 measured: {TT4_measured}")
    st.write(f"TT4: {TT4}")
    st.write(f"T4U measured: {T4U_measured}")
    st.write(f"T4U: {T4U}")
    st.write(f"FTI measured: {FTI_measured}")
    st.write(f"FTI: {FTI}")
    st.write(f"Referall source: {referral_source}")  


    # Convert categorical features to binary
    sex = 1 if sex == 'Male' else 0
    on_thyroxine = 1 if on_thyroxine == 'Yes' else 0
    query_on_thyroxine = 1 if query_on_thyroxine == 'Yes' else 0
    on_antithyroid_medication = 1 if on_antithyroid_medication == 'Yes' else 0
    sick = 1 if sick == 'Yes' else 0
    pregnant = 1 if pregnant == 'Yes' else 0
    thyroid_surgery = 1 if thyroid_surgery == 'Yes' else 0
    I131_treatment = 1 if I131_treatment == 'Yes' else 0
    query_hypothyroid = 1 if query_hypothyroid == 'Yes' else 0
    query_hyperthyroid = 1 if query_hyperthyroid == 'Yes' else 0
    lithium = 1 if lithium == 'Yes' else 0
    goitre = 1 if goitre == 'Yes' else 0
    tumor = 1 if tumor == 'Yes' else 0
    hypopituitary = 1 if hypopituitary == 'Yes' else 0
    psych = 1 if psych == 'Yes' else 0
    TSH_measured = 1 if TSH_measured == 'Yes' else 0
    T3_measured = 1 if T3_measured == 'Yes' else 0
    TT4_measured = 1 if TT4_measured == 'Yes' else 0
    T4U_measured = 1 if T4U_measured == 'Yes' else 0
    FTI_measured = 1 if FTI_measured == 'Yes' else 0

    # Create a dictionary to map class indices to labels
    class_labels = {0: 'Negative', 1: 'Primary hypothyroid', 2: 'Compensatory hypothyroid'}


    # Create a function to apply the models
    def predict_disease(features):
        # Scale the features
        features = scaler.transform(np.array(features).reshape(1, -1))
        # Predict the disease
        prediction = model.predict(features)
        # Decode the prediction
        prediction = lbl.inverse_transform(prediction)
        # Get prediction probabilities
        prediction_proba = model.predict_proba(features)
        return prediction, prediction_proba

    # Button to predict the disease
    if st.sidebar.button("Predict"):
        features = [age, sex, on_thyroxine, query_on_thyroxine, on_antithyroid_medication, sick, pregnant, thyroid_surgery, I131_treatment, query_hypothyroid, query_hyperthyroid,
        lithium, goitre, tumor, hypopituitary, psych, TSH_measured, TSH, T3_measured, T3, TT4_measured,
        TT4, T4U_measured, T4U, FTI_measured, FTI, referral_source]
        
    # Make prediction
        prediction, prediction_proba = predict_disease(features)
    
    # Map predicted class to label
        predicted_class = prediction[0]
        predicted_label = class_labels[predicted_class]

    # Display results
        st.write(f'The predicted disease class is: {prediction[0]}')
        st.write(f'Prediction Probabilities: {prediction_proba[0]}')
    
    # Display results - with class information
        st.write(f'The predicted disease class is: {predicted_label}')
        st.write(f'Prediction Probabilities: {prediction_proba[0]}')



if __name__ == '__main__':
    main()
