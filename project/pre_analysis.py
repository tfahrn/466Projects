
# coding: utf-8

# In[107]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sex = {"Erkek":1,"Kadın":0}
education = {"Lise":"High School","Lisans":"Bachelors Degree",
             "Ön Lisans":"Associates Degree",
            "Ortaokul":"Middle School","İlkokul":"Primary School",
            "Lisans Üstü":"Graduate"}
def conv_sex(row):
    return sex[row]

def conv_educ(row):
    return education[row]

def conv_qs(row):
    if row == "Evet":
        return 1
    else:
        return 0

def educ_graph(data):
    educ_counts = data[["Education","Timestamp"]].groupby("Education").count()
    educ_counts.rename(columns={"Timestamp":"count"},inplace=True)
    educ_counts.reset_index(inplace=True)

    plt.pie(educ_counts["count"],labels=educ_counts["Education"], autopct='%1.1f%%', shadow=True)
    plt.title("Education Levels")
    plt.savefig("ground_truth/educ_pie.jpg")
    plt.close()
    plt.clf()

def age_graph(data):
    age_counts = data[["Age","Timestamp"]].groupby("Age").count()
    age_counts.rename(columns={"Timestamp":"count"},inplace=True)
    age_counts.reset_index(inplace=True)

    plt.pie(age_counts["count"],labels=age_counts["Age"], autopct='%1.1f%%', shadow=True)
    plt.title("Age Groups")
    plt.savefig("ground_truth/age_pie.jpg")
    plt.close()
    plt.clf()

def party_graph(data):
    party_counts = data[["Party","Timestamp"]].groupby("Party").count()
    party_counts.rename(columns={"Timestamp":"count"},inplace=True)
    party_counts.reset_index(inplace=True)

    plt.pie(party_counts["count"], labels=party_counts['Party'], autopct='%1.1f%%', shadow=True)
    plt.title("Party Affiliation")
    plt.savefig("ground_truth/party_pie.jpg")
    plt.clf()
    
def region_graph(data):
    area_counts = data[["Area","Timestamp"]].groupby("Area").count()
    area_counts.rename(columns={"Timestamp":"count"},inplace=True)
    area_counts.reset_index(inplace=True)

    plt.pie(area_counts["count"], labels=area_counts['Area'], autopct='%1.1f%%', shadow=True)
    plt.title("Regions of Turkey")
    plt.savefig("ground_truth/regions_pie.jpg")
    plt.clf()
#def vectorize(): 


# In[105]:

colnames = ["Timestamp","Sex","Age","Area","Education","Q1","Q2","Q3","Q4"
           ,"Q5","Q6","Q7","Q8","Q9","Q10","Party"]
data = pd.read_csv("data/data.csv",names=colnames,skiprows=1)
data["Sex"] = data["Sex"].apply(conv_sex)
data["Education"] = data['Education'].apply(conv_educ)

#Graph ground truth for pre analysis
educ_graph(data)
age_graph(data)
region_graph(data)
party_graph(data)

#One hot encode are categorical variables
data = pd.get_dummies(data,prefix=["age","educ","area"],columns=["Age","Education","Area"])
data.drop("Timestamp",axis = 1, inplace = True)
questions = ["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8","Q9","Q10"]
for q in questions:
    data[q] = data[q].apply(conv_qs)


