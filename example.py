import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

def plot_data(x, y, k_class, iteration):
    plt.scatter(x, y, c=k_class)
    plt.title(f'K Means with k = {iteration}')
    plt.show(block=False)
    plt.pause(2)
    plt.close() 

def KMeans_model(data, k):
    kmeans = KMeans(n_clusters=k, n_init = 'auto')
    kmeans.fit(data)
    
    return kmeans

def inertia_graph(data):
    """
    In order to find the best value for K, we need to run K-means across our data for a range of possible values. 
    We only have 10 data points, so the maximum number of clusters is 10. So for each value K in range(1,11), 
    we train a K-means model and plot the intertia at that number of clusters:
    """ 
    inertias = []
    for i in range(1,11):
        kmeans_group = KMeans_model(data, i)
        inertias.append(kmeans_group.inertia_)
        plot_data(x, y, kmeans_group.labels_, i)

    plt.plot(range(1,11), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()


if __name__ == "__main__":
    #plot_data(x, y)
    data = list(zip(x, y))
    inertia_graph(data)
