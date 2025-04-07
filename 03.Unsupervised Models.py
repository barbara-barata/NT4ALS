import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D

# Load the data
file_path = "Baseline_Corrected_Spectra.xlsx"
df = pd.read_excel(file_path)

# Extract sample names and spectra
sample_names = df.iloc[:, 0]  # First column (Sample names)
spectra = df.iloc[:, 1:]      # Spectra data

# Load the data
file_path = "Samples group.xlsx"
df = pd.read_excel(file_path)

# Extract sample names and group
samples = df.iloc[:, 0]  # First column (Sample names)
sample_groups = df.iloc[:, 1:]      # Groups data

Comparing_samples = []
for i in range(0, len(spectra)):
    for j in range(0,len(sample_groups)):
        # Find equivalent samples
         if sample_names[i] == samples[j]:
             # If sample is "patient", add 1, otherwise, add 0
             if sample_groups.loc[i].Group == "Patient":
                Comparing_samples.append(1)
             elif sample_groups.loc[i].Group == "Control":
                Comparing_samples.append(0)
             else:
                 Comparing_samples.append(2)
    
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
# Define color mapping
group_colors = {1: "Blue", 0: "Red"}
group_labels = {1: "Patient", 0: "Control"}
fig, ax = plt.subplots(figsize=(8,6))

# Plot each group separately
for group, color in group_colors.items():
    mask = [c == group for c in Comparing_samples]
    ax.scatter(pca_df['PC1'][mask], pca_df['PC2'][mask], 
               color=color, label=group_labels[group], alpha=0.7)
    
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
plt.title('K-Means Clustering on PC1 vs PC2')
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

# Define color mapping
group_colors = {1: "Blue", 0: "Red"}
group_labels = {1: "Patient", 0: "Control"}

# 3D PCA plot results
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(projection='3d')
for group, color in group_colors.items():
    mask = [c == group for c in Comparing_samples]
    ax.scatter(pca_df['PC1'][mask], pca_df['PC2'][mask], pca_df['PC3'][mask], 
               color=color, label=group_labels[group], alpha=0.7)
ax.set_xlabel(f'PC1 ({explained_variance[0]:.2f}%)')
ax.set_ylabel(f'PC2 ({explained_variance[1]:.2f}%)')
ax.set_zlabel(f'PC3 ({explained_variance[2]:.2f}%)')
ax.set_title('3D PCA of FTIR Spectra')
ax.legend()
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
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

#PC2 vs PC3
plt.figure(figsize=(8,6))
for group, color in group_colors.items():
    mask = [c == group for c in Comparing_samples]
    plt.scatter(pca_df['PC2'][mask], pca_df['PC3'][mask], 
                color=color, label=group_labels[group], alpha=0.7)

plt.xlabel(f'PC2 ({explained_variance[1]:.2f}%)')
plt.ylabel(f'PC3 ({explained_variance[2]:.2f}%)')
plt.title('PCA of FTIR Spectra (PC2 vs. PC3)')
plt.legend()
plt.show()

#K-means PC2 vsPC3
plt.figure(figsize=(8,6))
sns.scatterplot(x=pca_df['PC2'], y=pca_df['PC3'], hue=pca_df['Cluster'], palette='viridis', s=100)
plt.scatter(centroids[:, 1], centroids[:, 2], c='red', marker='X', s=200, label='Centroids')  # Mark centroids
plt.xlabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
plt.ylabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.2f}%)')
plt.title('K-Means Clustering on PC2 vs PC3')
plt.legend()
plt.show()

#PC1 vs PC3
plt.figure(figsize=(8,6))
for group, color in group_colors.items():
    mask = [c == group for c in Comparing_samples]
    plt.scatter(pca_df['PC1'][mask], pca_df['PC3'][mask], 
                color=color, label=group_labels[group], alpha=0.7)

plt.xlabel(f'PC1 ({explained_variance[0]:.2f}%)')
plt.ylabel(f'PC3 ({explained_variance[2]:.2f}%)')
plt.title('PCA of FTIR Spectra (PC1 vs. PC3)')
plt.legend()
plt.show()

#K-means PC1 vsPC3
plt.figure(figsize=(8,6))
sns.scatterplot(x=pca_df['PC1'], y=pca_df['PC3'], hue=pca_df['Cluster'], palette='viridis', s=100)
plt.scatter(centroids[:, 0], centroids[:, 2], c='red', marker='X', s=200, label='Centroids')  # Mark centroids
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.2f}%)')
plt.title('K-Means Clustering on PC1 vs PC3')
plt.legend()
plt.show()

#PLS-DA Model
# Convert Comparing_samples to numpy array
y = np.array(Comparing_samples)

# Standardize the spectral data (same as before)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(spectra)

# Apply PLS-DA with 3 components
pls = PLSRegression(n_components=3)
pls.fit(X_scaled, y)
pls_components = pls.transform(X_scaled)  # Get PLS scores

# Create a DataFrame with PLS results
pls_df = pd.DataFrame(data=pls_components, columns=['PLS1', 'PLS2', 'PLS3'])
pls_df['Sample'] = sample_names
pls_df['Group'] = y  # Add sample labels

# Plot PLS-DA results
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(projection='3d')
for group, color in group_colors.items():
    mask = pls_df['Group'] == group
    ax.scatter(pls_df['PLS1'][mask], pls_df['PLS2'][mask], pls_df['PLS3'][mask], 
               color=color, label=group_labels[group], alpha=0.7)
ax.set_xlabel('PLS1')
ax.set_ylabel('PLS2')
ax.set_zlabel('PLS3')
ax.set_title('3D PLS-DA of FTIR Spectra')
ax.legend()
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

#PLS1 vs PLS2
plt.figure(figsize=(8,6))
for group, color in group_colors.items():
    mask = pls_df['Group'] == group
    plt.scatter(pls_df['PLS1'][mask], pls_df['PLS2'][mask], color=color, label=group_labels[group], alpha=0.7)
plt.xlabel('PLS1')
plt.ylabel('PLS2')
plt.title('PLS-DA of FTIR Spectra (PLS1 vs. PLS2)')
plt.legend()
plt.show()

#PLS2 vs PLS3
plt.figure(figsize=(8,6))
for group, color in group_colors.items():
    mask = pls_df['Group'] == group
    plt.scatter(pls_df['PLS2'][mask], pls_df['PLS3'][mask], color=color, label=group_labels[group], alpha=0.7)

plt.xlabel('PLS2')
plt.ylabel('PLS3')
plt.title('PLS-DA of FTIR Spectra (PLS2 vs. PLS3)')
plt.legend()
plt.show()

#PLS1 vs PLS3
plt.figure(figsize=(8,6))
for group, color in group_colors.items():
    mask = pls_df['Group'] == group
    plt.scatter(pls_df['PLS1'][mask], pls_df['PLS3'][mask], color=color, label=group_labels[group], alpha=0.7)

plt.xlabel('PLS1')
plt.ylabel('PLS3')
plt.title('PLS-DA of FTIR Spectra (PLS1 vs. PLS3)')
plt.legend()
plt.show()



