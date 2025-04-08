import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import zscore

# Importa as funções do módulo plot_utils
from zz_plots_func import plot_2d_scatter, plot_3d_scatter

# Define os mapeamentos de cores e labels para os grupos
group_colors = {1: "blue", 0: "red", 2: "gray"}
group_labels = {1: "Patient", 0: "Control", 2: "Unknown"}

# Carregar dados espectrais e grupos usando merge
df_spectra = pd.read_excel("Baseline_Corrected_Spectra.xlsx")
df_groups  = pd.read_excel("Samples group.xlsx")
df = pd.merge(df_spectra, df_groups, left_on=df_spectra.columns[0],
              right_on=df_groups.columns[0], how='left')

# Mapear os grupos: "Patient" -> 1, "Control" -> 0;
group_map = {"Patient": 1, "Control": 0}
df['Group'] = df['Group'].map(group_map).fillna(2).astype(int)

sample_names = df.iloc[:, 0]
spectra = df.iloc[:, 1:df_spectra.shape[1]]
# todo: dar debug aqui
# @bb este array tem 1101 entradas para cada pessoa, o que não faz sentido uma vez que era suposto já ter
# havido uma parte do espectro que deveria ter sido cortada
# tens de me explicar como é que isso acontece.
# e isso faz-me querer ver os epectros tratados antes de entrarem aqui. os plots da baseline removal

print (spectra)

# Remover outliers via z-score
z_scores = np.abs(zscore(spectra, nan_policy='omit'))
mask = ~(np.any(z_scores >= 10, axis=1))
spectra_clean = spectra[mask]
sample_names_clean = sample_names[mask]
groups_clean = df['Group'][mask].to_numpy()
# todo: dar count do número de outliers
num_outliers = len(spectra) - len(spectra_clean)
print(f"Número de outliers removidos: {num_outliers}")

# Padronização
scaler = StandardScaler()
scaled_spectra = scaler.fit_transform(spectra_clean)

# PCA
pca = PCA(n_components=3)
pcs = pca.fit_transform(scaled_spectra)
explained = pca.explained_variance_ratio_ * 100

# DataFrame do PCA (utilizado para visualizar os resultados do PCA)
pca_df = pd.DataFrame(pcs, columns=['PC1', 'PC2', 'PC3'])
pca_df['Group'] = groups_clean

print(f"Explained Variance: PC1={explained[0]:.2f}%, PC2={explained[1]:.2f}%, PC3={explained[2]:.2f}%")

# Plot 2D do PCA (PC1 vs. PC2)
plot_2d_scatter(pca_df, 'PC1', 'PC2', 'PCA (PC1 vs. PC2)',
                group_colors=group_colors, group_labels=group_labels,
                xlabel=f'PC1 ({explained[0]:.2f}%)', ylabel=f'PC2 ({explained[1]:.2f}%)',
                legend_kwargs={'bbox_to_anchor': (1, 1)})

# Plot 3D do PCA (PC1, PC2 e PC3)
plot_3d_scatter(pca_df, 'PC1', 'PC2', 'PC3', '3D PCA of FTIR Spectra',
                group_colors=group_colors, group_labels=group_labels,
                xlabel=f'PC1 ({explained[0]:.2f}%)', ylabel=f'PC2 ({explained[1]:.2f}%)', zlabel=f'PC3 ({explained[2]:.2f}%)',
                legend_kwargs={'loc': 'upper left', 'bbox_to_anchor': (1.05, 1)})

# PLS-DA com dados escalados (3 componentes)
pls = PLSRegression(n_components=3)
pls_components = pls.fit_transform(scaled_spectra, groups_clean)[0]
pls_df = pd.DataFrame(pls_components, columns=['PLS1', 'PLS2', 'PLS3'])
pls_df['Group'] = groups_clean

# Plot 2D do PLS-DA (PLS1 vs. PLS2)
plot_2d_scatter(pls_df, 'PLS1', 'PLS2', 'PLS-DA (PLS1 vs. PLS2)',
                group_colors=group_colors, group_labels=group_labels)

# Plot 3D do PLS-DA (PLS1, PLS2 e PLS3)
plot_3d_scatter(pls_df, 'PLS1', 'PLS2', 'PLS3', '3D PLS-DA of FTIR Spectra',
                group_colors=group_colors, group_labels=group_labels,
                xlabel='PLS1', ylabel='PLS2', zlabel='PLS3',
                legend_kwargs={'loc': 'upper left', 'bbox_to_anchor': (1.05, 1)})

# PLS-DA usando dados transformados pelo PCA (3 componentes)
pls_pca = PLSRegression(n_components=3)
pls_pca_components = pls_pca.fit_transform(pcs, groups_clean)[0]
pls_pca_df = pd.DataFrame(pls_pca_components, columns=['PLS1', 'PLS2', 'PLS3'])
pls_pca_df['Group'] = groups_clean

# Plot 2D do PLS-DA PCA (PLS1 vs. PLS2)
plot_2d_scatter(pls_pca_df, 'PLS1', 'PLS2', 'PLS-DA PCA (PLS1 vs. PLS2)',
                group_colors=group_colors, group_labels=group_labels)

# Plot 3D do PLS-DA PCA (PLS1, PLs2 e PLS3)
plot_3d_scatter(pls_pca_df, 'PLS1', 'PLS2', 'PLS3', '3D PLS-DA PCA of FTIR Spectra',
                group_colors=group_colors, group_labels=group_labels,
                xlabel='PLS1', ylabel='PLS2', zlabel='PLS3',
                legend_kwargs={'loc': 'upper left', 'bbox_to_anchor': (1.05, 1)})


