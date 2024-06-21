import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from pycaret.classification import *
from sklearn.model_selection import train_test_split
from pycaret.classification import setup
import time

# Page title
st.set_page_config(page_title='ElderGuard- Message Fraud detector for the elderly', page_icon='👴🏼👵🏾📱')
st.title('👴🏼👵🏾📱 ElderGuard- Message Fraud detector for the elderly')

with st.expander('About this project: "Why the elderly?"'):
  st.markdown('**How bad is this issue?**')
  st.warning('''
1) In 2023, the FBI's Internet Crime Complaint Center (IC3) total losses reported to the IC3 
by those over the age of 60 topped $3.4 billion, an almost 11% increase in
reported losses from 2022.
2) In Spain, the number of fraud complaints to the Bank of Spain doubled in 2022, reaching 10,361 cases of fraud.
3) In Spain, one in three people over 65 has been a victim of at least one cybercrime.
''' )

  st.markdown('**What is our target segment?**')
  st.info('''
Our targeted users represent the elderly with +65 years of age , who balance technology use with
traditional banking needs and have significant interaction with family and community activities.

Tech-Savvy and Non-Tech savvy users are considered, given that even with sufficient knowledge, anyone can be vulnerable 
while having their guard down. 

Look at Ramón Artal, 71 years old and resident of Sant Feliu de Llobregat (Barcelona), 
who has a background in engineering, is a tech-savvy person and has been close to falling victim to a digital scam
  ''')

st.markdown('**Our solution**')
st.info('''
Our value proposition is clear: elders and small banks/insurers can significantly reduce fraud-related losses and
enhance customer trust by implementing our fraud detection algorithm. This solution is cost-
effective and easy to integrate, offering robust protection against fraudulent transactions.
''' )

 # st.markdown('**Under the hood**')
 # st.markdown('Data sets:')
 # st.code('''- Drug solubility data set
 # ''', language='markdown')
  
 # st.markdown('Libraries used:')
 # st.code('''- Pandas for data wrangling
#- Scikit-learn for building a machine learning model
#- Altair for chart creation
#- Streamlit for user interface
#  ''', language='markdown')

st.markdown('*Try it out yourself!*')
input_message= st.text_input('Message')
st.markdown('**Result**')

sleep_time= 1

# Initiate the model building process
if input_message: 
    with st.status("Running ...", expanded=True) as status:
        st.write("Loading data ...")
        time.sleep(sleep_time)

        st.write("Cleaning text: tokenizing, removing punctuations and removing stopwords ...")
        time.sleep(sleep_time)
        # Descargar stopwords y tokenizer de NLTK
        nltk.download('stopwords')
        nltk.download('punkt')

        def clean_text(text):
            # Remove punctuations and convert to lowercase
            text = re.sub(r'[^\w\s]', '', text.lower())

            # Tokenize the text
            tokens = word_tokenize(text)

            # Remove stopwords
            tokens = [word for word in tokens if word not in stopwords.words('english')]

            return ' '.join(tokens)

        # Load the column names from the file
        with open('column_names.txt', 'r') as file:
            column_names = [line.strip() for line in file]

        # Clean the email message
        cleaned_message = clean_text(input_message)

        # Create a DataFrame from the cleaned message
        data = {'Message': [cleaned_message]}
        df = pd.DataFrame(data)
            
        st.write("Inserting in word database (Using Count Vectorizer)...")
        #Applying Count Vectorizer
        cv = CountVectorizer(stop_words='english')
        dtm = cv.fit_transform(df['Message'])

        # Convert the document-term matrix to a DataFrame for better visualization
        dtm_df = pd.DataFrame(dtm.toarray(), columns=cv.get_feature_names_out())

        # Ensure new_dtm_df has the same columns as the original training dataset
        for col in column_names:
            if col not in dtm_df.columns:
                dtm_df[col] = 0

        # Reorder columns to match the original training dataset
        dtm_df = dtm_df[column_names]

        st.write("Loading machine learning model ...")
        # Load the model
        Fraud_Detector = load_model('Msg_Fraud_Detector')
        
        st.write("Applying model to make predictions ...")
        # Make predictions
        predictions = predict_model(Fraud_Detector, data=dtm_df)

    status.update(label="Status", state="complete", expanded=False)

    if predictions['prediction_label'].iloc[0] == 'fraud':
        st.markdown("🚨 **ALERT: This email has a HIGH probability of being fraudulent.**🚨")
        st.markdown("""
        **Suggested Action:**
        1. Do not click on any links or download any attachments in the message.
        2. Do not provide any personal information or financial details.
        3. Delete the message from your inbox.
        """)
    else:
        st.markdown("✅ **This email is probably safe, but remember to always check the following:**")
        st.markdown("""
        **Safety Reminders:**
        1. Always verify the sender's email address.
        2. Look for signs of phishing, such as poor grammar or suspicious links.
        3. Be cautious of unsolicited messages requesting personal or financial information.
        """)

    moreinfo = st.radio("Do you want to run other tests just to be completely sure?", ("Yes", "No"))
    if moreinfo == 'Yes':
        # Display a radio button widget
        st.markdown("**If it still seems suspicious, let's check some other aspects:**")
        selection1 = st.radio("Does this message have an official link? For example: ups.com (If they claim they are from UPS)", ("Yes", "No"))

        # Use the selected option
        if selection1 == "Yes":
            st.write("✅ Great, It seems to be safe!")
        else:
            st.write("🚨 If it is not an official link, we would not advise clicking it. It is better to manually search any information in the official website by yourself if needed.")
        
        selection2 = st.radio("Does this message have an official number? For example: You can check UPS's official number in Google to see if it matches this message", ("Yes", "No"))
        # Use the selected option
        if selection2 == "Yes":
            st.write("✅ Great, It seems to be safe!")
        else:
            st.write("🚨 If it is not an official number, we would advise calling official numbers by yourself if needed.")
    else:
        st.markdown("✅ **Perfect! Feel free to continue using our model!**")


# Ask for text input if none is entered
else:
    st.warning('Enter a message to see the prediction!')

    
# Ask for CSV upload if none is detected
else:
    st.warning('👈 Upload a CSV file or click *"Load example data"* to get started!')
