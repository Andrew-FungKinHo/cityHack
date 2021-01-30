import pandas as pd
import numpy as np
pd.options.display.max_columns = 50
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_data(filename): 
    df = pd.read_csv('Covid Dataset.csv')

    df.replace('Yes', 1, inplace = True)
    df.replace('No', 0, inplace = True)

    trash_features = ['Wearing Masks', 'Sanitization from Market']

    df.drop(trash_features, axis = 1, inplace = True)

    result = df['COVID-19']

    symptoms = df.drop('COVID-19', axis = 1)
    features = symptoms.columns.tolist()
    #user_input.append(input_data)

    symptoms_train, symptoms_test, result_train, result_test = train_test_split(symptoms, result, random_state = 69, shuffle = True)

    model = RandomForestClassifier(n_estimators = 200, random_state = 42)
    model.fit(symptoms_train, result_train)
    return model, features


# user_input = []
# input_data = []

# for i in features: 
#     state = False
#     while state == False: 
#         option = input('Do you have/ have experienced: ' + i + '  ')
#         if (option == "Y" or option == "y"): 
#             state = True
#             input_data.append(1)
#         elif (option == "N" or option == "n"):
#             state = True
#             input_data.append(0)
#         else:
#             print("Invalid")

# if (len(input_data) != len(features)):
#     print('Error')
 
def test(model, user_input): 
    input_data = []
    input_data.append(user_input)
    user_proba = model.predict_proba(input_data)

    neg_prob = user_proba[0][0]*100
    pos_prob = user_proba[0][1]*100
    return pos_prob, neg_prob

def answer_conversion(user_input):
    new_list  = [1 if input == "Yes" else 0 for input in user_input]
    return new_list

# print("The prob that you have covid is: ", pos_prob, "% :D")