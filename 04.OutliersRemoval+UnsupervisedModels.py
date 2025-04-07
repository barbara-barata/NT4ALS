import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import IsolationForest
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import zscore

# Load spectral data
spectra_path = "Baseline_Corrected_Spectra.xlsx"
df_spectra = pd.read_excel(spectra_path)
sample_names = df_spectra.iloc[:, 0]
spectra = df_spectra.iloc[:, 1:]

# Load sample groups
groups_path = "Samples group.xlsx"
df_groups = pd.read_excel(groups_path)
samples = df_groups.iloc[:, 0]
sample_groups = df_groups.iloc[:, 1]

# Match sample groups
Comparing_samples = []
for i in range(len(sample_names)):
    matched_group = df_groups.loc[df_groups.iloc[:, 0] == sample_names[i], "Group"].values
    if len(matched_group) > 0:
        if matched_group[0] == "Patient":
            Comparing_samples.append(1)
        elif matched_group[0] == "Control":
            Comparing_samples.append(0)
        else:
            Comparing_samples.append(2)
    else:
        Comparing_samples.append(2)  # Default to unknown if no match

y = np.array(Comparing_samples)

#Compute z-score for entire spectra
z_scores = np.abs(zscore(spectra, nan_policy='omit'))
outlier_mask = np.any(z_scores >= 5, axis=1)
outlier_count = np.sum(outlier_mask)
spectra_clean = spectra[~outlier_mask]
sample_names_clean = sample_names[~outlier_mask]
y_clean = y[~outlier_mask]


# Apply Isolation Forest for outlier detection
#iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
#iso_forest.fit(spectra)
#outliers = iso_forest.predict(spectra)  # -1 = outlier, 1 = normal

# Keep only normal data
#spectra_clean = spectra[outliers == 1]
#sample_names_clean = sample_names[outliers == 1]
#y_clean = y[outliers == 1]

# Standardize the cleaned data
scaler = StandardScaler()
scaled_spectra = scaler.fit_transform(spectra_clean)

# Apply PCA
pca = PCA(n_components=3)
principal_components = pca.fit_transform(scaled_spectra)
pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2', 'PC3'])
pca_df['Group'] = y_clean

# Variance explained by each component (%)
explained_variance = pca.explained_variance_ratio_ * 100
print(f"Explained Variance: PC1={explained_variance[0]:.2f}%, PC2={explained_variance[1]:.2f}%, PC3={explained_variance[2]:.2f}%")

# Define color mapping
group_colors = {1: "blue", 0: "red"}
group_labels = {1: "Patient", 0: "Control"}

# 3D PCA Plot
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(projection='3d')
for group, color in group_colors.items():
    mask = pca_df['Group'] == group
    ax.scatter(pca_df['PC1'][mask], pca_df['PC2'][mask], pca_df['PC3'][mask], 
               color=color, label=group_labels[group], alpha=0.7)
ax.set_xlabel(f'PC1 ({explained_variance[0]:.2f}%)')
ax.set_ylabel(f'PC2 ({explained_variance[1]:.2f}%)')
ax.set_zlabel(f'PC3 ({explained_variance[2]:.2f}%)')
ax.set_title('3D PCA of FTIR Spectra')
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

# PC1 vs PC2
plt.figure(figsize=(8,6))
for group, color in group_colors.items():
    mask = pca_df['Group'] == group
    plt.scatter(pca_df['PC1'][mask], pca_df['PC2'][mask], color=color, label=group_labels[group], alpha=0.7)
plt.xlabel(f'PC1 ({explained_variance[0]:.2f}%)')
plt.ylabel(f'PC2 ({explained_variance[1]:.2f}%)')
plt.title('PCA of FTIR Spectra (PC1 vs. PC2)')
plt.legend(bbox_to_anchor=(1,1))
plt.show()

# PC2 vs PC3
plt.figure(figsize=(8,6))
for group, color in group_colors.items():
    mask = pca_df['Group'] == group
    plt.scatter(pca_df['PC2'][mask], pca_df['PC3'][mask], color=color, label=group_labels[group], alpha=0.7)
plt.xlabel(f'PC2 ({explained_variance[0]:.2f}%)')
plt.ylabel(f'PC3 ({explained_variance[1]:.2f}%)')
plt.title('PCA of FTIR Spectra (PC2 vs. PC3)')
plt.legend(bbox_to_anchor=(1,1))
plt.show()

# PC1 vs PC3
plt.figure(figsize=(8,6))
for group, color in group_colors.items():
    mask = pca_df['Group'] == group
    plt.scatter(pca_df['PC1'][mask], pca_df['PC3'][mask], color=color, label=group_labels[group], alpha=0.7)
plt.xlabel(f'PC1 ({explained_variance[0]:.2f}%)')
plt.ylabel(f'PC3 ({explained_variance[1]:.2f}%)')
plt.title('PCA of FTIR Spectra (PC1 vs. PC3)')
plt.legend(bbox_to_anchor=(1,1))
plt.show()

# K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
pca_df['Cluster'] = kmeans.fit_predict(principal_components)

# Apply PLS-DA
pls = PLSRegression(n_components=3)
pls.fit(scaled_spectra, y_clean)
pls_components = pls.transform(scaled_spectra)
pls_df = pd.DataFrame(pls_components, columns=['PLS1', 'PLS2', 'PLS3'])
pls_df['Group'] = y_clean

# 3D PLS-DA Plot
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

# Apply PLS-DA using PCA-transformed data
pls_pca = PLSRegression(n_components=3)
pls_pca.fit(principal_components, y_clean)
pls_pca_components = pls_pca.transform(principal_components)

# Convert to DataFrame for plotting
pls_pca_df = pd.DataFrame(pls_pca_components, columns=['PLS1', 'PLS2', 'PLS3'])
pls_pca_df['Group'] = y_clean

# 3D PLS-DA Plot (using PCA data)
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(projection='3d')
for group, color in group_colors.items():
    mask = pls_pca_df['Group'] == group
    ax.scatter(pls_pca_df['PLS1'][mask], pls_pca_df['PLS2'][mask], pls_pca_df['PLS3'][mask], 
               color=color, label=group_labels[group], alpha=0.7)
ax.set_xlabel('PLS1')
ax.set_ylabel('PLS2')
ax.set_zlabel('PLS3')
ax.set_title('PLS-DA using PCA-transformed Data')
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

# PLS1 vs PLS2
plt.figure(figsize=(8,6))
for group, color in group_colors.items():
    mask = pls_pca_df['Group'] == group
    plt.scatter(pls_pca_df['PLS1'][mask], pls_pca_df['PLS2'][mask], color=color, label=group_labels[group], alpha=0.7)
plt.xlabel('PLS1')
plt.ylabel('PLS2')
plt.title('PLS-DA (PLS1 vs. PLS2)')
plt.legend()
plt.show()

# PLS2 vs PLS3
plt.figure(figsize=(8,6))
for group, color in group_colors.items():
    mask = pls_pca_df['Group'] == group
    plt.scatter(pls_pca_df['PLS2'][mask], pls_pca_df['PLS3'][mask], color=color, label=group_labels[group], alpha=0.7)
plt.xlabel('PLS2')
plt.ylabel('PLS3')
plt.title('PLS-DA (PLS2 vs. PLS3)')
plt.legend()
plt.show()

# PLS1 vs PLS3
plt.figure(figsize=(8,6))
for group, color in group_colors.items():
    mask = pls_pca_df['Group'] == group
    plt.scatter(pls_pca_df['PLS1'][mask], pls_pca_df['PLS3'][mask], color=color, label=group_labels[group], alpha=0.7)
plt.xlabel('PLS1')
plt.ylabel('PLS3')
plt.title('PLS-DA (PLS1 vs. PLS3)')
plt.legend()
plt.show()