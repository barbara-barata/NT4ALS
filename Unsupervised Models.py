import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Load the data
file_path = "C:/Users/bsbar/NT4ALS/Baseline_Corrected_Spectra.xlsx"
df = pd.read_excel(file_path)

# Extract sample names and spectra
sample_names = df.iloc[:, 0]  # First column (Sample names)
spectra = df.iloc[:, 1:]      # Spectra data

# Standardize the data (important for PCA)
scaler = StandardScaler()
scaled_spectra = scaler.fit_transform(spectra)

# Apply PCA
pca = PCA(n_components=2)  # Use 2 principal components
principal_components = pca.fit_transform(scaled_spectra)

# Create a dataframe with PCA results
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Sample'] = sample_names

# Variance explained by each component (%)
explained_variance = pca.explained_variance_ratio_ * 100
print(f"Explained Variance: PC1={explained_variance[0]:.2f}%, PC2={explained_variance[1]:.2f}%")

#Bar Graph results
num_components = len(explained_variance)
x_labels = [f'PC{i+1}' for i in range(num_components)]

plt.figure(figsize=(8, 5))
plt.bar(x_labels, explained_variance, color='#1f77b4')  # Use the values directly
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance (%)')
plt.title('Explained Variance of Each Principal Component')
plt.ylim(0, max(explained_variance) + 5)  # Adjust limit for better visualization
plt.show()

# Plot PCA results
plt.figure(figsize=(8,6))
sns.scatterplot(x=pca_df['PC1'], y=pca_df['PC2'], hue=pca_df['Sample'], palette='viridis', s=100)
plt.xlabel(f'PC1 ({explained_variance[0]:.2f}%)')
plt.ylabel(f'PC2 ({explained_variance[1]:.2f}%)')
plt.title('PCA of FTIR Spectra')
plt.legend(bbox_to_anchor=(1,1))
plt.show()

# Choose the number of clusters (k)
k = 2  # Change this based on the dataset

# Apply K-Means
kmeans = KMeans(n_clusters=k)
pca_df['Cluster'] = kmeans.fit_predict(principal_components)  # Assign clusters

# Get cluster centers in PCA space
centroids = kmeans.cluster_centers_

plt.figure(figsize=(8,6))
sns.scatterplot(x=pca_df['PC1'], y=pca_df['PC2'], hue=pca_df['Cluster'], palette='viridis', s=100)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')  # Mark centroids
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
plt.title('K-Means Clustering on PCA-Reduced FTIR Data')
plt.legend()
plt.show()

# Apply PCA with 3PC
pca = PCA(n_components=3)  # Use 3 principal components
principal_components = pca.fit_transform(scaled_spectra)

# Create a dataframe with PCA results
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
pca_df['Sample'] = sample_names

# Variance explained by each component (%)
explained_variance = pca.explained_variance_ratio_ * 100
print(f"Explained Variance: PC1={explained_variance[0]:.2f}%, PC2={explained_variance[1]:.2f}%, PC3={explained_variance[2]:.2f}%")

#Bar Graph results
num_components = len(explained_variance)
x_labels = [f'PC{i+1}' for i in range(num_components)]

plt.figure(figsize=(8, 5))
plt.bar(x_labels, explained_variance, color='#1f77b4')  # Use the values directly
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance (%)')
plt.title('Explained Variance of Each Principal Component')
plt.ylim(0, max(explained_variance) + 5)  # Adjust limit for better visualization
plt.show()

# Plot PCA results
#plt.figure(figsize=(8,6))
#sns.scatterplot(x=pca_df['PC1'], y=pca_df['PC2'], hue=pca_df['Sample'], palette='viridis', s=100)
#plt.xlabel(f'PC1 ({explained_variance[0]:.2f}%)')
#plt.ylabel(f'PC2 ({explained_variance[1]:.2f}%)')
#plt.title('PCA of FTIR Spectra')
#plt.legend(bbox_to_anchor=(1,1))
#plt.show()

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(projection='3d')
for i in range (0,len(pca_df)):
    ax.scatter(pca_df['PC1'][i], pca_df['PC2'][i], pca_df['PC3'][i], label=pca_df["Sample"][i])
ax.legend(bbox_to_anchor=(1,1))
plt.show()

# Choose the number of clusters (k)
k = 2  # Change this based on the dataset

# Apply K-Means
kmeans = KMeans(n_clusters=k)
pca_df['Cluster'] = kmeans.fit_predict(principal_components)  # Assign clusters

# Get cluster centers in PCA space
centroids = kmeans.cluster_centers_

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(projection='3d')
for i in range (0,len(pca_df)):
    color = "red"
    if pca_df["Cluster"][i]==0:
        color = "blue"
    elif pca_df["Cluster"][i]==1:
        color = "yellow"
    ax.scatter(pca_df['PC1'][i], pca_df['PC2'][i], pca_df['PC3'][i], color=color)
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', marker='X', s=200, label='Centroids')
ax.legend(bbox_to_anchor=(1,1))
plt.show()





