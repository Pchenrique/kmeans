import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Getting data from file
dados = np.genfromtxt("colorrectal_2_classes_formatted.txt", delimiter=",")

# Getting classes
classes = dados[:, 0:142]
attributes = dados
# Getting attributes and normalizing
#attributes = np.delete(dados,(142), axis=1)
min_max_scaler = MinMaxScaler()
attributes_norm = min_max_scaler.fit_transform(attributes)

kmeans = KMeans(n_clusters = 2, init = 'random')

kmeans.fit(attributes_norm)

labels = kmeans.labels_

plt.scatter(attributes_norm[:, 0], attributes_norm[:,1], s = 100, c = kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'red',label = 'Centroids')
plt.title('Colorrectal Clusters and Centroids')
plt.xlabel('SepalLength')
plt.ylabel('SepalWidth')
plt.legend()

plt.show()


