import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import csv

csv_data = "resources/xor.csv"

def show_data(x, y, color):
    plt.scatter(x, y, c=color)
    plt.show()

def KMeans_model(data, k):
    kmeans = KMeans(n_clusters=k, n_init = 'auto')
    kmeans.fit(data)
    
    return kmeans

def inertia_graph(data):

    inertias = []
    for i in range(1,len(data)):
        kmeans_group = KMeans_model(data, i)
        inertias.append(kmeans_group.inertia_)
    inertias[0] = 0
    plt.plot(range(1,len(data)), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

def main():
    with open(csv_data, "r") as file:
        header = file.readline()
        csv_reader = csv.reader(file)
        data_list = []
        x = []
        y = []
        for row in csv_reader:
            if (row[0] == "C1"):
                x.append(float(row[1]))
                y.append(float(row[2]))
    
    train_data = list(zip(x, y))
    inertia_graph(train_data)
    show_data(x, y, KMeans_model(train_data, 2).labels_)

if __name__ == "__main__":
    main()
