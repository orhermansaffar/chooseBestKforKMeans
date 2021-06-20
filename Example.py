# choose features
data_for_clustering = data[features_for_clustering].copy()
data_for_clustering.fillna(0,inplace=True)
# create data matrix
data_matrix = np.matrix(data_for_clustering).astype(float)
# scale the data
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
scaled_data = mms.fit_transform(data_matrix)
# choose k range
k_range=range(2,20)
best_k, results = chooseBestKforKMeansParallel(scaled_data, k_range)
# plot the results
plt.figure(figsize=(7,4))
plt.plot(results,'o')
plt.title('Adjusted Inertia for each K')
plt.xlabel('K')
plt.ylabel('Adjusted Inertia')
plt.xticks(range(2,20,1))
