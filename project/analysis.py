import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kmeans
import agglomerative
sex = {"Erkek":"Male","Kadın":"Female"}
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
    
def educ_cluster_graph(data,cluster,t):
    educ_counts = data[["Education","Timestamp"]].groupby("Education").count()
    educ_counts.rename(columns={"Timestamp":"count"},inplace=True)
    educ_counts.reset_index(inplace=True)

    plt.pie(educ_counts["count"],labels=educ_counts["Education"], autopct='%1.1f%%', shadow=True)
    plt.title(t+"-Education Levels for Cluster: "+ str(cluster))
    plt.savefig("results/educ_pie_c"+str(cluster)+t+".jpg")
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
    
def age_cluster_graph(data,cluster,t):
    age_counts = data[["Age","Timestamp"]].groupby("Age").count()
    age_counts.rename(columns={"Timestamp":"count"},inplace=True)
    age_counts.reset_index(inplace=True)
    
    plt.pie(age_counts["count"],labels=age_counts["Age"], autopct='%1.1f%%', shadow=True)
    plt.title(t+"-Age Groups for Cluster: "+str(cluster))
    plt.savefig("results/age_pie_c"+str(cluster)+t+".jpg")
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
    
def party_cluster_graph(data,cluster,t):
    party_counts = data[["Party","Timestamp"]].groupby("Party").count()
    party_counts.rename(columns={"Timestamp":"count"},inplace=True)
    party_counts.reset_index(inplace=True)

    plt.pie(party_counts["count"], labels=party_counts['Party'], autopct='%1.1f%%', shadow=True)
    plt.title(t+"-Party Affiliation for Cluster: " +str(cluster))
    plt.savefig("results/party_pie_c"+str(cluster)+t+".jpg")
    plt.clf()
    
def region_graph(data):
    area_counts = data[["Area","Timestamp"]].groupby("Area").count()
    area_counts.rename(columns={"Timestamp":"count"},inplace=True)
    area_counts.reset_index(inplace=True)

    plt.pie(area_counts["count"], labels=area_counts['Area'], autopct='%1.1f%%', shadow=True)
    plt.title("Regions of Turkey")
    plt.savefig("ground_truth/regions_pie.jpg")
    plt.clf()

def sex_cluster_graph(data,cluster,t):
    sex_counts = data[["Sex","Timestamp"]].groupby("Sex").count()
    sex_counts.rename(columns={"Timestamp":"count"},inplace=True)
    sex_counts.reset_index(inplace=True)

    plt.pie(sex_counts["count"], labels=sex_counts['Sex'], autopct='%1.1f%%', shadow=True)
    plt.title(t+"-Sex-Cluster: " +str(cluster))
    plt.savefig("results/sex_pie_c"+str(cluster)+t+".jpg")
    plt.clf()
    
def sex_graph(data):
    sex_counts = data[["Sex","Timestamp"]].groupby("Sex").count()
    sex_counts.rename(columns={"Timestamp":"count"},inplace=True)
    sex_counts.reset_index(inplace=True)

    plt.pie(sex_counts["count"], labels=sex_counts['Sex'], autopct='%1.1f%%', shadow=True)
    plt.title("Sex")
    plt.savefig("ground_truth/sex_pie.jpg")
    plt.clf()
    
    
def region_cluster_graph(data,cluster,t):
    area_counts = data[["Area","Timestamp"]].groupby("Area").count()
    area_counts.rename(columns={"Timestamp":"count"},inplace=True)
    area_counts.reset_index(inplace=True)

    plt.pie(area_counts["count"], labels=area_counts['Area'], autopct='%1.1f%%', shadow=True)
    plt.title(t+"Regions of Turkey for Cluster: " + str(cluster))
    plt.savefig("results/regions_pie_c"+str(cluster)+t+".jpg")
    plt.clf()

def questions_graph(q_ratios,t):
    index = np.arange(1,11)
    plt.plot(index,q_ratios[0],color="blue",label="cluster 0")
    plt.scatter(index,q_ratios[0],color="blue")
    plt.plot(index,q_ratios[1],color="red",label = "cluster 1")
    plt.scatter(index,q_ratios[1],color="red")
    plt.legend(loc = "upper left")
    plt.xlabel("Questions")
    plt.ylabel("Ratio of Yes/No")
    plt.title(t+"-Ratio of Yes/No vs Question")
    plt.savefig("results/questions_"+t+".jpg")
    plt.clf()

def get_data():
    colnames = ["Timestamp","Sex","Age","Area","Education","Q1","Q2","Q3","Q4"
           ,"Q5","Q6","Q7","Q8","Q9","Q10","Party"]
    data = pd.read_csv("data/data.csv",names=colnames,skiprows=1)
    data["Sex"] = data["Sex"].apply(conv_sex)
    data["Education"] = data['Education'].apply(conv_educ)
    return data

def pre_analysis():
    data = get_data()
    #Graph ground truth for pre analysis
    educ_graph(data)
    age_graph(data)
    region_graph(data)
    party_graph(data)
    sex_graph(data)
    
def q_numeric(data):
    questions = ["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8","Q9","Q10"]

    for q in questions:
        data[q] = data[q].apply(conv_qs)
def run_kmean_stats():
    data = get_data()
    '''
    #One hot encode are categorical variables
    data = pd.get_dummies(data,prefix=["age","educ","area"],columns=["Age","Education","Area"])
    data.drop("Timestamp",axis = 1, inplace = True)
    '''
    q_numeric(data)

    clusters = kmeans.main_p(2)
    data["cluster"] = clusters
    #Runs graphs for given clustering method
    q_ratios = []
    questions = ["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8","Q9","Q10"]
    for i in list(set(clusters)):
        sub_data = data[data['cluster'] == i]
        age_cluster_graph(sub_data,i,"kmeans")
        party_cluster_graph(sub_data,i,"kmeans")
        educ_cluster_graph(sub_data,i,"kmeans")
        region_cluster_graph(sub_data,i,"kmeans")
        sex_cluster_graph(sub_data,i,"kmeans")

        yes_to_no = []
        n = len(sub_data)
        for q in questions:
            yes_to_no.append(np.sum(sub_data[q])/(n - np.sum(sub_data[q])))
        q_ratios.append(yes_to_no)
    questions_graph(q_ratios,"kmeans")
    
def run_agglom_stats():
    data = get_data()
    '''
    #One hot encode are categorical variables
    data = pd.get_dummies(data,prefix=["age","educ","area"],columns=["Age","Education","Area"])
    data.drop("Timestamp",axis = 1, inplace = True)
    '''
    q_numeric(data)

    clusters = agglomerative.main_p(2)
    data["cluster"] = clusters
    #Runs graphs for given clustering method
    q_ratios = []
    questions = ["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8","Q9","Q10"]
    for i in list(set(clusters)):
        sub_data = data[data['cluster'] == i]
        age_cluster_graph(sub_data,i,"agglom")
        party_cluster_graph(sub_data,i,"agglom")
        educ_cluster_graph(sub_data,i,"agglom")
        region_cluster_graph(sub_data,i,"agglom")
        sex_cluster_graph(sub_data,i,"agglom")

        yes_to_no = []
        n = len(sub_data)
        for q in questions:
            if(n - np.sum(sub_data[q])==0):
                yes_to_no.append(np.sum(sub_data[q])/1)
            else:
                yes_to_no.append(np.sum(sub_data[q])/(n - np.sum(sub_data[q])))
        q_ratios.append(yes_to_no)
    questions_graph(q_ratios,"agglom")
    
def main():
    pre_analysis()
    run_kmean_stats()
    run_agglom_stats()

if __name__ == '__main__':
    main()
