import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import zscore

# Importa as funções do módulo plot_utils
from zz_plots_func import plot_2d_scatter, plot_3d_scatter

# Define os mapeamentos de cores e labels para os grupos
group_colors = {1: "blue", 0: "red"}
group_labels = {1: "Patient", 0: "Control"}

# Carregar dados espectrais e grupos usando merge
df_spectra = pd.read_excel("Normalized_Spectra.xlsx")
df_groups  = pd.read_excel("Samples group.xlsx")
df = pd.merge(df_spectra, df_groups, left_on=df_spectra.columns[0],
              right_on=df_groups.columns[0], how='left')

# Mapear os grupos: "Patient" -> 1, "Control" -> 0;
group_map = {"Patient": 1, "Control": 0}
df['Group'] = df['Group'].map(group_map).fillna(2).astype(int)

sample_names = df.iloc[:, 0]
spectra = df.iloc[:, 1:df_spectra.shape[1]]
  
print (spectra)

# Remover outliers via z-score
z_scores = np.abs(zscore(spectra, nan_policy='omit'))
mask = ~(np.any(z_scores >= 5, axis=1))
spectra_clean = spectra[mask]
sample_names_clean = sample_names[mask]
groups_clean = df['Group'][mask].to_numpy()
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

# Plot 2D do PCA
for i in range(1,4):
    for j in range(1,4):
        if j>i:
            plot_2d_scatter(pca_df, f'PC{i}', f'PC{j}', f'PCA (PC{i} vs. PC{j})',
                group_colors=group_colors, group_labels=group_labels,
                xlabel=f'PC{i} ({explained[i-1]:.2f}%)', ylabel=f'PC{j}({explained[j-1]:.2f}%)',
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

# Plot 2D do PLS-DA
for i in range(1,4):
    for j in range(1,4):
        if j>i:
            plot_2d_scatter(pls_df, f'PLS{i}', f'PLS{j}', f'PLS-DA (PLS{i} vs. PLS{j})',
                group_colors=group_colors, group_labels=group_labels)

# Plot 3D do PLS-DA (PLS1, PLS2 e PLS3)
plot_3d_scatter(pls_df, 'PLS1', 'PLS2', 'PLS3', '3D PLS-DA of FTIR Spectra',
                group_colors=group_colors, group_labels=group_labels,
                xlabel='PLS1', ylabel='PLS2', zlabel='PLS3',
                legend_kwargs={'loc': 'upper left', 'bbox_to_anchor': (1.05, 1)})

#PCA para o sexo
# Define os grupos por sexo
group_colors = {1: "blue", 0: "red"}
group_labels = {1: "Female", 0: "Male"}

# Mapear os grupos
group_map = {"Feminino": 1, "Masculino": 0}
df['Sex'] = df['Sex'].map(group_map).fillna(2).astype(int)

sample_names = df.iloc[:, 0]
spectra = df.iloc[:, 1:df_spectra.shape[1]]
  
print (spectra)

sex_clean = df['Sex'][mask].to_numpy()

#PCA plot
pca = PCA(n_components=3)
pcs = pca.fit_transform(scaled_spectra)
explained = pca.explained_variance_ratio_ * 100

pca_df = pd.DataFrame(pcs, columns=['PC1', 'PC2', 'PC3'])
pca_df['Group'] = sex_clean

print(f"Explained Variance: PC1={explained[0]:.2f}%, PC2={explained[1]:.2f}%, PC3={explained[2]:.2f}%")

for i in range(1,4):
    for j in range(1,4):
        if j>i:
            plot_2d_scatter(pca_df, f'PC{i}', f'PC{j}', f'PCA (PC{i} vs. PC{j}) by sex',
                group_colors=group_colors, group_labels=group_labels,
                xlabel=f'PC{i} ({explained[i-1]:.2f}%)', ylabel=f'PC{j}({explained[j-1]:.2f}%)',
                legend_kwargs={'bbox_to_anchor': (1, 1)})
            
# PLS-DA para o sexo
pls = PLSRegression(n_components=3)
pls_components = pls.fit_transform(scaled_spectra, groups_clean)[0]
pls_df = pd.DataFrame(pls_components, columns=['PLS1', 'PLS2', 'PLS3'])
pls_df['Group'] = sex_clean

# Plot 2D do PLS-DA
for i in range(1,4):
    for j in range(1,4):
        if j>i:
            plot_2d_scatter(pls_df, f'PLS{i}', f'PLS{j}', f'PLS-DA (PLS{i} vs. PLS{j}) by sex',
                group_colors=group_colors, group_labels=group_labels)
            
#PCA para a idade
# Define os grupos por idade
for i in range (len(df['Age'])):
        if df['Age'].iloc[i] >= 30 and df['Age'].iloc[i]<50:
            df['Age'].iloc[i] = 1
        elif df['Age'].iloc[i] >= 50 and df['Age'].iloc[i]<70:
            df['Age'].iloc[i] = 0
        elif df['Age'].iloc[i] >= 70 and df['Age'].iloc[i]<=100:
            df['Age'].iloc[i] = 2

group_colors = {1: "blue", 0: "red", 2: "yellow"}
group_labels = {1: "[30-50[", 0: "[50-70[", 2: "[70-100]"}

age_clean = df['Age'][mask].to_numpy()

#PCA plot
pca = PCA(n_components=3)
pcs = pca.fit_transform(scaled_spectra)
explained = pca.explained_variance_ratio_ * 100

pca_df = pd.DataFrame(pcs, columns=['PC1', 'PC2', 'PC3'])
pca_df['Group'] = age_clean

print(f"Explained Variance: PC1={explained[0]:.2f}%, PC2={explained[1]:.2f}%, PC3={explained[2]:.2f}%")

for i in range(1,4):
    for j in range(1,4):
        if j>i:
            plot_2d_scatter(pca_df, f'PC{i}', f'PC{j}', f'PCA (PC{i} vs. PC{j}) by age',
                group_colors=group_colors, group_labels=group_labels,
                xlabel=f'PC{i} ({explained[i-1]:.2f}%)', ylabel=f'PC{j}({explained[j-1]:.2f}%)',
                legend_kwargs={'bbox_to_anchor': (1, 1)})
            
# PLS-DA para a idade
pls = PLSRegression(n_components=3)
pls_components = pls.fit_transform(scaled_spectra, groups_clean)[0]
pls_df = pd.DataFrame(pls_components, columns=['PLS1', 'PLS2', 'PLS3'])
pls_df['Group'] = age_clean

# PLS-DA plot
for i in range(1,4):
    for j in range(1,4):
        if j>i:
            plot_2d_scatter(pls_df, f'PLS{i}', f'PLS{j}', f'PLS-DA (PLS{i} vs. PLS{j}) by age',
                group_colors=group_colors, group_labels=group_labels)
