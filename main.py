import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import json
from openai import OpenAI

client = OpenAI(base_url="https://api.groq.com/openai/v1",
                api_key=os.environ.get('GROQ_API_KEY'))

url = "https://churn-ml-models-114a.onrender.com/predict"


def prepare_input(credit_score, location, gender, age, tenure, balance,
                  num_products, has_credit_card, is_active_member,
                  estimated_salary):

    input_dict = {
        'CreditScore': credit_score,
        'Age': age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCreditCard": has_credit_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary,
        "CLV": balance * estimated_salary / 100000,
        "Middle-Aged": 1 if (30 < age < 45) else 0,
        "Senior": 1 if (45 < age < 60) else 0,
        "Elderly": 1 if (60 < age < 100) else 0,
        "Geography_France": 1 if location == "France" else 0,
        "Geography_Germany": 1 if location == "Germany" else 0,
        "Geography_Spain": 1 if location == "Spain" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
    }
    return input_dict


def list_results(data):

    avg_prediction = int(np.mean(list(data['prediction'].values())))
    
    st.markdown("### Model Predictions")
    for model, pred in data['prediction'].items():
        st.write(f"- {model}: {pred}")
    st.write(f"Average Prediction: {avg_prediction}")
    
    avg_probability = np.mean(list(data['probability'].values()))
    
    st.markdown("### Model Probabilities")
    for model, prob in data['probability'].items():
        st.write(f"- {model}: {prob}")
    st.write(f"Average Probability: {avg_probability}")
    
    return avg_prediction, avg_probability



def explain_prediction(probability, input_dict, surname):
    prompt = f"""You are an expert data scientist at a bank, specializing in interpreting and explaining customer churn predictions. The machine learning model predicts that the customer, {surname}, has a {round(probability * 100, 1)}% chance of churning, based on the following details:

Customer Information:
{input_dict}

You will carefully consider that the age in the given customer info falls into one of these categories:
    Young: Age < 30
    Middle-Aged: 30 < Age < 45
    Senior: 45 < Age < 60
    Elderly: 60 < Age < 100

For example, if the customer is 40 years old, you should explain that they are middle-aged. Another example, if the customer is 56 years old, you should explain that they are senior.

Top 10 Key Factors Influencing Churn Risk:

    Feature                | Importance
    -------------------------------------
    AgeGroup_Senior         | 0.359508
    NumOfProducts           | 0.112505
    IsActiveMember          | 0.078493
    Age                     | 0.051761
    Geography_Germany       | 0.043180
    Gender_Male             | 0.022967
    Balance                 | 0.021839
    CLV                     | 0.020594
    Geography_Spain         | 0.017027
    EstimatedSalary         | 0.013020

Below are summary statistics for customers who churned:
{df[df['Exited'] == 1].describe()}

You will provide a clear, concise explanation of the customer's likelihood of churning based on their individual details and the provided key features. 

- If the customer's risk of churning is greater than 40%, explain in three sentences why they may be at risk of churning.

- If the customer's risk of churning is less than 25%, explain in three sentences why they may not be at significant risk.

The explanation should reference the customer's information, relevant feature importance, and general trends from churned customers. Avoid mentioning the churn probability, model predictions, or technical terms such as 'machine learning models' or 'top 10 features.' Instead, directly explain the prediction in a natural, human-friendly manner. Do not mention the features of the model by name, for example, "The age falls into the 'Middle-Age' category 30<Age<45.

 You will keep the explanation concise and limied to three sentences.
"""

    print("EXPLANATION PROMPT", prompt)

    raw_response = client.chat.completions.create(model="llama-3.2-3b-preview",
                                                  messages=[{
                                                      "role": "user",
                                                      "content": prompt
                                                  }])
    return raw_response.choices[0].message.content


def generate_email(probability, input_dict, explanation, surname):
    prompt = f"""You are a manager at HS Bank. You are responsible for 
    ensuring customers stay with the bank and are incentivized with various      offers.

    You noticed a customer's information:
    {input_dict}

    Here is some explanation as to why the customer might be at risk of churning:
    {explanation}

    Generate an email to the customer based on their information, asking them to stay if they are at risk of churning, or offerering them incentives so that they become more loyal to the bank.

    Make sure to list out a set of incentives to stay based on their information, in bullet point format. Don't ever mention the probability of churning, or the machine learning model to the customer.

    """
    raw_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": prompt
        }],
    )

    print("\n\nEMAIL PROMPT", prompt)

    return raw_response.choices[0].message.content


st.title("Customer Churn Prediction")

df = pd.read_csv("churn.csv")

customers = [
    f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()
]

selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
    selected_customer_id = int(selected_customer_option.split(" - ")[0])
    # print("Selected Customer ID", selected_customer_id)

    selected_surname = selected_customer_option.split(" - ")[1]
    # print("Surname", selected_surname)

    selected_customer = df[df["CustomerId"] == selected_customer_id].iloc[0]

    # print("Selected Customer", selected_customer)

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score",
                                       min_value=300,
                                       max_value=850,
                                       value=int(
                                           selected_customer["CreditScore"]))

        location = st.selectbox("Location", ["Spain", "France", "Germany"],
                                index=["Spain", "France", "Germany"
                                       ].index(selected_customer["Geography"]))

        gender = st.radio(
            "Gender", ["Male", "Female"],
            index=0 if selected_customer["Gender"] == 'Male' else 1)

        age = st.number_input("Age",
                              min_value=18,
                              max_value=100,
                              value=int(selected_customer["Age"]))

        tenure = st.number_input("Tenure",
                                 min_value=0,
                                 max_value=50,
                                 value=int(selected_customer["Tenure"]))

    with col2:
        balance = st.number_input("Balance",
                                  min_value=0.0,
                                  value=float(selected_customer["Balance"]))

        num_products = st.number_input("Number of Products",
                                       min_value=1,
                                       max_value=10,
                                       value=int(
                                           selected_customer["NumOfProducts"]))

        has_credit_card = st.checkbox("Has Credit Card",
                                      value=bool(
                                          selected_customer["HasCrCard"]))

        is_active_member = st.checkbox(
            "Is Active Member",
            value=bool(selected_customer["IsActiveMember"]))

        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=float(selected_customer["EstimatedSalary"]))

    input_dict = prepare_input(credit_score, location, gender, age, tenure,
                               balance, num_products, has_credit_card,
                               is_active_member, estimated_salary)

    response = requests.post(url, json=input_dict)

    if response.status_code == 200:
        result = response.json()

        avg_prediction, avg_probability = list_results(result)

        explanation = explain_prediction(avg_probability, input_dict,
                                         selected_customer["Surname"])

        st.markdown("---")

        st.subheader('Explanation of Prediction')

        st.markdown(explanation)

        email = generate_email(avg_probability, input_dict, explanation,
                               selected_customer["Surname"])

        st.markdown("---")

        st.subheader("Personalized Email")

        st.markdown(email)

    else:
        print("Error:", response.status_code, response.text)
