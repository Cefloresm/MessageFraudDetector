import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from pycaret.classification import setup
import time

# Page title
st.set_page_config(page_title='ElderGuard- Message Fraud detector for the elderly', page_icon='👴🏼👵🏾📱')
st.title('👴🏼👵🏾📱 ElderGuard- Message Fraud detector for the elderly')

with st.sidebar:
    st.markdown('**Here are some typical fraud messages for you to test this with:**')
    st.code('''Dear Friend, I am Mr. Ben Suleman a custom officer and work as Assistant controller of the Customs and Excise department Of the Federal Ministry of Internal Affairs stationed at the Murtala Mohammed International Airport, Ikeja, Lagos-Nigeria. After the sudden death of the former Head of state of Nigeria General Sanni Abacha on June 8th 1998 his aides and immediate members of his family were arrested while trying to escape from Nigeria in a Chartered jet to Saudi Arabia with 6 trunk boxes Marked "Diplomatic Baggage". Acting on a tip-off as they attempted to board the Air Craft,my officials carried out a thorough search on the air craft and discovered that the 6 trunk boxes contained foreign currencies amounting to US$197,570,000.00(One Hundred and Ninety-Seven Million Five Hundred Seventy Thousand United States Dollars). I declared only (5) five boxes to the government and withheld one (1) in my custody containing the sum of (US$30,000,000.00) Thirty Million United States Dollars Only, which has been disguised to prevent their being discovered during transportation process.Due to several media reports on the late head of state about all the money him and his co-government officials stole from our government treasury amounting to US$55 Billion Dollars (ref:ngrguardiannews.com) of July 2nd 1999. Even the London times of July 1998 reported that General Abacha has over US$3.Billion dollars in one account overseas. We decided to conceal this one (1)box till the situation is calm and quite on the issue. The box was thus deposited with a security company here in Nigeria and tagged as "Precious Stones and Jewellry" in other that its content will not be discovered. Now that all is calm, we (myself and two of my colleagues in the operations team) are now ready to move this box out of the country through a diplomatic arrangement which is the safest means. However as government officials the Civil Service Code of Conduct does not allow us by law to operate any foreign account or own foreign investment and the amount of money that can be found in our account cannot be more than our salary on the average, thus our handicapp and our need for your assistance to help collect and keep safely in your account this money. Therefore we want you to assist us in moving this money out of Nigeria. We shall definitely compensate you handsomely for the assistance. We can do this by instructing the Security Company here in Nigeria to move the consignment to their affiliate branch office outside Nigeria through diplomatic means and the consignment will be termed as Precious Stones and Jewelleries" which you bought during your visit to Nigeria and is being transfered to your country from here for safe keeping. Then we can arrange to meet at the destination country to take the delivery of the consignment. You will thereafter open an account there and lodge the Money there and gradually instruct remittance to your Country. This business is 100% risk free for you so please treat this matter with utmost confidentiality .If you indicate your interest to assist us please just e-mail me for more Explanation on how we plan to execute the transaction. Expecting your response urgently. Best regards, Mr. Ben Suleman
    ''')
    st.code('Apple: Congrats! Your IP address has been chosen for this months Apple free product tester! Confirm shipping address here dgeo7.com/Z4ygZpmn1k to claim your prize')
    st.markdown('Here is one that is a fraud but the algorithm says it is true. This is a real message I was sent several months ago.')
    st.code('USPS: the arranged delivery for the shipment 1z81183 has been changed. Please confirm here: w7fzc.info/fTYntliDhl')
    
    st.markdown('Even though it is marked as probably safe, there are still some manual check recommendations provided. Read further below after analysis is done.')     
    



with st.expander('About this project: "Why the elderly?"'):
  st.info('**How bad is this issue?**')
  st.warning('''
1) In 2023, the FBI's Internet Crime Complaint Center (IC3) total losses reported to the IC3 
by those over the age of 60 topped **$3.4 billion**, an almost 11% increase in
reported losses from 2022.
2) In Spain, the number of fraud complaints to the Bank of Spain doubled in 2022, reaching 10,361 cases of fraud.
3) In Spain, one in three people over 65 has been a victim of at least one cybercrime.
''' )

  st.info('**What is our target segment?**')
  st.markdown('''
Our targeted users represent the elderly with +65 years of age , who balance technology use with
traditional banking needs and have significant interaction with family and community activities.

Tech-Savvy and Non-Tech savvy users are considered, given that even with sufficient knowledge, anyone can be vulnerable 
while having their guard down. 

Look at Ramón Artal, 71 years old and resident of Sant Feliu de Llobregat (Barcelona), 
who has a background in engineering, is a tech-savvy person and has been close to falling victim to a digital scam
  ''')

st.info('**Our solution**')
st.markdown('''
Our value proposition is clear: elders and small banks/insurers can significantly reduce cyber-related losses and
enhance customer trust by implementing our fraud detection algorithm. This solution would be cost-
effective and easy to integrate, offering robust protection against fraudulent text messages/e-mails and eventually calls.
''' )


st.markdown('*Try it out yourself!*')
st.warning('There are some examples in the sidebar to test it with. Click on the top-left corner of the page to find them. Hover over each text and click the copy button.')
input_message= st.text_area('Message (press Cmd+ Return to load in Macbook)')
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

    st.warning('**Want to know how it works?!** 🚀')
    st.markdown('''
    Our fraud detection system is like having your tech-savvy family member by your side, always ready to spot something fishy. 
    Here’s how it works:
    ''')

    st.markdown('''
      **1. Cleaning Text** 🧹
      ''')
    
    st.markdown('''
      First, we tidy up the text data to make it easier to analyze. This involves (but is not limited to):
      ''')
    
    st.markdown('''
        • Tokenizing: Breaking sentences into individual words.
	''')
    
    st.markdown('''
        • Removing Punctuations: Getting rid of commas, periods, and other punctuation marks.
	''')
    
    st.markdown('''
        • Removing Stopwords: Eliminating common words like “and”, “the”, “is” that are not relevant for fraud analysis.
	''')

    st.image('textclean.jpeg', caption="Text cleaning example: Lemmatization")

	

    st.markdown('''
      **2. Counting Word Frequency** 🔢
      ''')
    
    st.markdown('''
      Next, we process these cleaned words using a method called “Bag of Words (BoW)”, used to count how often each word appears in the text.
      ''')
    
    st.image('BoW.png',caption="Bag of Words Example")
    
    
    st.markdown('''
      **3. Analyzing Fraud or Non-Fraud with trained Machine Learning Model (Logistic Regression)** ⚙️
      ''')
    
    st.markdown('''
      Finally, a trained supervised machine learning model (Logistic Regression) analyses the BoW and estimates the probability of fraud/no fraud based on the words inputted.

      This logistic regression model was trained with a dataset of +10k rows of real fraudulent messages. During training, the model learned what combinations of words are likely to be a fraud, for example, suspicious words or links to untrustworthy websites.

      Once trained, it can analyze new messages by outputting a probability score between 0% and 100%. The closer to 100% it is, the more likely it is a fraud message. 

      Below is a graph showing an example of what words (or "features") the fraud detection algorithm considers as most important when determining if it is a fraud or not:
      ''')
    
    st.image('featureimportance.jpg', caption="Feature Importance Graph")

    


# Ask for text input if none is entered
else:
    st.warning('Enter a message to see the prediction!')

st.info('**FAQ**')
st.markdown('''
**1.** **How would user data be protected with this fraud detector** **?**

User data protection is a top priority for this solution. The product would employ data encryption and adhere to local data regulations such as the General Data Protection Regulation (GDPR) as principal regulatory framework in the E.U. and United Kingdom, which has also been an inspiration for future statutes to be created by nations like the U.S. With GDPR, “individuals effectively own their personal information and thus presumptively have the legal right to control it, and who can use it is a matter for them to decide (Reuters, 2023)” 

For the present application, this means that users would have the possibility to decide how their data is used.
It is important to remark that allowing the use of their data means that the algorithm can make their lives safer by identifying and learning from harmful attempts of fraud (a nice trade-off for a better service).


**2.** **Why is the prediction “not fraud” if it is actually a fraudulent message** **?**

Like all models, the one used in this scenario is not 100% accurate and can be further improved in the next prototype iterations with what we could call “the obvious”, which are no-sense checks that can be spotted by anyone paying attention, e.g.:
- A person claiming to be a customer support agent of private bank but with a “@gmail.com” email address.
- A parcel delivery provider with a different number than the one found on official websites.

In this first iteration, these no-sense checks are reminded to the user even though the prediction is "no fraud" to ensure maximum protection.


**3.** **How would users receive alerts if fraudulent activity is detected** **?**

Alerts would be sent directly to the user’s device or both the user and a designated caregiver, depending on the setup. These alerts can be received via email or SMS, providing immediate information about potential fraud.




'''
)

st.info('**Next steps**')
st.markdown('''
1. Debugging and improve model functionality by including no-sense checks mentioned above.

2. Testing the economic viability of the solution. How to monetize it? Who would pay for the solution: users to directly avoid the risk or insurance companies to reduce their no. of claims and therefore their loss ratio? 

3. Testing the technical feasibility of rolling out this solution to users.
''' )

with st.expander('**Some links used for research**'):
  st.markdown('''

  https://www.munichre.com/en/insights/cyber/cyber-insurance-risks-and-trends-2024.html

  https://www.reuters.com/legal/legalindustry/us-data-privacy-laws-enter-new-era-2023-2023-01-12/
  
  '''
  ) 
