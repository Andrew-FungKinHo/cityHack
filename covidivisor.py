import streamlit as st 
import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import Feedback
import Model
import pandas as pd
from PIL import Image



def main():
    st.set_page_config(page_title='COVIDvisor')
    # st.title('COVIDvisor')
    sidebarTitle = st.sidebar.title('COVIDvisor')
    
    menu = ['Home', 'Test', 'View Data', 'About']
    page = st.sidebar.selectbox(
        'Main Menu',
        ('Home', 'Test', 'View Data', 'About')
    )
    if page == "Test":
        st.title('Feeling sick?')
        rand_forest, features = Model.train_data('Covid Dataset.csv')

        st.write(""" 
            ## Input your symptoms and get a preliminary diagnosis.
        """)
        

        dataset_name = st.sidebar.selectbox(
            'Select Illness to test for',
            ('COVID-19', 'Common Cold', 'Flu')
        )

        st.write(f"### Currently testing for {dataset_name} ")

        

        breathingAnswer= st.selectbox( 'Do you have any breathing problems?', ('-----','Yes', 'No',))
        feverAnswer= st.selectbox( 'Do you have a Fever (body temperature above 38˚C) ?', ('-----','Yes', 'No',))
        dryCoughAnswer= st.selectbox( 'Do you a dry cough?', ('-----','Yes', 'No',))
        soreThroatAnswer= st.selectbox( 'Do you a sore throat?', ('-----','Yes', 'No',))
        runningNoseAnswer = st.selectbox( 'Do you a running nose?', ('-----','Yes', 'No',))
        asthmaAnswer = st.selectbox( 'Do you have asthma?', ('-----','Yes', 'No',))
        chronicLungDiseaseAnswer = st.selectbox( 'Do you have chronic Lung Disease?', ('-----','Yes', 'No',))
        headacheAnswer = st.selectbox( 'Have you experienced headache?', ('-----','Yes', 'No',))
        heartDiseaseAnswer = st.selectbox( 'Do you have heart Disease?', ('-----','Yes', 'No',))
        diabetesAnswer = st.selectbox( 'Do you have diabetes?', ('-----','Yes', 'No',))
        hyperTensionAnswer = st.selectbox( 'Have your experienced any hyper Tension?', ('-----','Yes', 'No',))
        fatigueAnswer = st.selectbox( 'Are you fatigued last couple days?', ('-----','Yes', 'No',))
        gastrointestinalAnswer = st.selectbox( 'Do you have Gastrointestinal Disease?', ('-----','Yes', 'No',))
        abroadTravelAnswer = st.selectbox( 'Have you travelled abroad?', ('-----','Yes', 'No',))
        contactWithCOVIDPatientAnswer = st.selectbox( 'Were you in close contact with another COVID-19 postive patient last couple weeks?', ('-----','Yes', 'No',))
        attendedLargeGatheringAnswer = st.selectbox( 'Did you attend any large gathering last couple weeks?', ('-----','Yes', 'No',))
        visitedPublicExposedPlacesAnswer = st.selectbox( 'Have you visited any public exposed places?', ('-----','Yes', 'No',))
        familyWorkingInPublicExposedPlacesAnswer = st.selectbox( 'Do you have any family members that are currently working in publicly exposed places?', ('-----','Yes', 'No',))


        user_answers = [breathingAnswer,feverAnswer,dryCoughAnswer,soreThroatAnswer,runningNoseAnswer,asthmaAnswer,chronicLungDiseaseAnswer,headacheAnswer,heartDiseaseAnswer,diabetesAnswer,hyperTensionAnswer,fatigueAnswer,gastrointestinalAnswer,abroadTravelAnswer,contactWithCOVIDPatientAnswer,attendedLargeGatheringAnswer,visitedPublicExposedPlacesAnswer,familyWorkingInPublicExposedPlacesAnswer]

        if st.button("Get Checked"):
            
            page = "Home"
            st.success(f'Results submitted. Go to View Data page for detailed report')
            pos, neg = Model.test(rand_forest, Model.answer_conversion(user_answers))
            suggestedFeedback = Feedback.feedback(pos)
            st.write(f'You have {int(pos)} % of catching the Novel Coronavirus (COVID-19)')
            st.write(f'Feedback: {suggestedFeedback}')
            but = st.button("View Detail Results")
            # if but:
            #     page = 'View Data'
            # if st.button("Check your detailed report"):

    elif page == 'Home':
        st.title('COVIDvisor')
        
        st.write(""" 
            ## dedicated to providing an accurate pre-screening and initial diagnosis service for users with common illnesses in order to minimise unnecessary social contacts and the pressure of public hospitals.
            """)
        image = Image.open('doctor_image.jpg')
        st.image(image, use_column_width=True)
        st.write('Our solution is a machine learning (web) application that provides users with an accurate preliminary diagnosis given their input of symptoms currently experienced. The algorithm will suggest 3 top illnesses with the highest likelihood and for each illness, also giving the proper treatment methods and guidelines. ')
        

    elif page == 'View Data':
        st.title('Your Result')
        col1,col2 = st.beta_columns(2)
        with col1:
            st.header('What should I do?')
            st.write('''
                Avoid all social contact immediately to prevent spreading the virus to others.
                If your symptoms are severe and you are feeling unwell, please visit AE in hospital. Otherwise,
                test for COVID in one of the following locations. In the meantime, stay hydrated and get enough rest,
                and be sure to notify your family and friends.
            ''')
            
            st.subheader('Key symptoms you had related to COVID-19')
            st.write('**Fever**')
            st.write('''
                Fever is the most common symptom of COVID-19. 
                Seek medical advice and enforce social distancing if the conditions continue.
            ''')
            st.write('**Dry Cough**')
            st.write('''
                Studies have found that at least 60\%\of people with COVID-19 have a dry cough. Stay vigilant if you are having
                these symptoms.
            ''')
            st.write('**Contact with COVID-19 Patient**')
            st.write('''
                COVID-19 can spread from an infected person's mouth or nose in small liquid particles when
                they cough, sneeze, speak, sing or breathe heavily.
            ''')


        with col2:
            st.header('Your risk level is **92%**:')
            riskLevel = st.select_slider(
                '',
            options=['Low', 'Medium','High'],value='High')

            st.write('Your risk level of getting COVID-19 is **', riskLevel, '**')

            st.button('Consult a doctor')
            st.write('**CONDUCT A THOROUGH COVID-19 TEST AT THE NEAREST TEST CENTER**')

            # insert map 
            data = [[22.336241357823056, 114.16659858285237]] 
            centerDF = pd.DataFrame(data, columns=['lat', 'lon'])
            st.map(centerDF)

            st.write('Closest Test Centre: **Pak Tin Community Hall in Sham Shui Po**')

            st.write('**Address**: Pak Tin Estate, Pak Tin Street, Shek Kip Mei, Sham Shui Po, Kowloon')
            st.write('**Contractor**: Hong Kong Molecular Pathology Diagnostic Centre Limited')
            st.write('**Hotline**: 3611 1301 / 3611 1302')
            st.write('**Email**: COVID_Enquiry@hk-mpdc.com')
    else:
        st.title('Further development')
        st.write('Our application \'s position is to detect common illnesses as a starting point. However, with our application proven to be an effective tool for pre-screening / preliminary diagnosis, we can partner with the Hong Kong Government\'s  eHealth (醫健通) programme to allow users to import / link their medical history, further increasing the accuracy of the diagnosis results')


if __name__ == "__main__":
    main()
# st.write(f'Classifier = Random Forest')
# st.write(f'Answer = {user_answers}')

# pos, neg = Model.test(rand_forest, [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# # suggestedFeedback = Feedback.feedback(pos)

# st.write(f'You have : {pos} % of catching the Novel Coronavirus')

# st.write(f'Feedback: {suggestedFeedback}')