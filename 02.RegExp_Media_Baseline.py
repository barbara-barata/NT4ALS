import pandas as pd
import re
from BaselineRemoval import BaselineRemoval
import numpy as np
import matplotlib.pyplot as plt
import ast


# Substituir 'caminho_do_arquivo' pelo caminho real do teu arquivo Excel
caminho_do_arquivo = 'RAW_DATA.xlsx'
data = pd.read_excel(caminho_do_arquivo)

# Função para extrair o 'sample code'
def extrair_sample_code(nome):
    match = re.search(r'URINA_([A-Z]\d+\.\d+)', nome)
    if match:
        return match.group(1)
    match = re.search(r'URINE_([A-Z]\d+\.\d+)', nome)
    if match:
        return match.group(1)
    return "Código Inválido" # Retorna isso se o padrão não for encontrado

# Aplicar a função para transformar a primeira coluna em 'sample code'
data.iloc[:, 0] = data.iloc[:, 0].apply(extrair_sample_code)

# Descartar apenas a segunda coluna
data = data.drop(data.columns[1], axis=1)

# Imprimir as primeiras linhas do DataFrame processado
print(data.head())
print ("second part starts here!!")

# Definindo a coluna 'NAM' como o nome da amostra
data.rename(columns={data.columns[0]: 'NAM'}, inplace=True)

# Agrupar por 'NAM' e calcular a média dos espectros
data_agrupada = data.groupby('NAM').mean()

# Imprimir as primeiras linhas do DataFrame processado
print(data_agrupada.head())

# Aplicar a correção de baseline a cada linha
def corrigir_baseline(espectro):
    baseObj = BaselineRemoval(espectro)
    return baseObj.ModPoly(2)  # O número 2 define o grau do polinômio

# Aplicar a correção a todas as linhas
baseline_corrected_spectral_data = data_agrupada.apply(corrigir_baseline, axis=1)

# Converter para DataFrame
baseline_corrected_spectral_data_df = pd.DataFrame(baseline_corrected_spectral_data)

# Expanding the array into multiple columns
baseline_corrected_spectral_data_df_expanded = baseline_corrected_spectral_data_df.iloc[:,0].apply(pd.Series)

# Renaming columns (optional)
baseline_corrected_spectral_data_df_expanded.columns = [f'col_{i}' for i in range(baseline_corrected_spectral_data_df_expanded.shape[1])]

print(baseline_corrected_spectral_data_df_expanded)

# Salvar os resultados corrigidos
baseline_corrected_spectral_data_df_expanded.to_excel("Baseline_Corrected_Spectra.xlsx", index=True)

print("Correção de baseline concluída e salva em 'Baseline_Corrected_Spectra.xlsx'.")


def euclidean_normalization(spectrum):
    euclidean_length = np.linalg.norm(spectrum)
    return spectrum / euclidean_length

normalized_spectral_data = baseline_corrected_spectral_data.copy()

# Apply Euclidean normalization to each spectrum
for i in range(0, len(baseline_corrected_spectral_data)):
    normalized_spectral_data[i] = np.apply_along_axis(euclidean_normalization, axis=0, arr=baseline_corrected_spectral_data[i])

# Converter para DataFrame
normalized_spectral_data_df = pd.DataFrame(normalized_spectral_data)

# Expanding the array into multiple columns
normalized_spectral_data_df_expanded = normalized_spectral_data_df.iloc[:,0].apply(pd.Series)

# Renaming columns (optional)
normalized_spectral_data_df_expanded.columns = [f'col_{i}' for i in range(normalized_spectral_data_df_expanded.shape[1])]

print(normalized_spectral_data_df_expanded)
            
#Eliminar pontos do espectro no intervalo 1850-2500
index_mask = []
for k in range(0, len(data_agrupada.columns)):
    if (not(1850 <= data_agrupada.columns[k] <= 2500)):
        index_mask.append(1)
    else:
        index_mask.append(0)
               
bool_mask = np.array(index_mask).astype(bool)

normalized_spectral_data_df_expanded_filtered = normalized_spectral_data_df_expanded.loc[:,bool_mask]
cut_columns = data_agrupada.columns[bool_mask]
cut_columns_first = []
cut_columns_second = []
for j in range(0,len(cut_columns)):
    if cut_columns[j] <= 1850:
        cut_columns_first.append(cut_columns[j])
    else:
        cut_columns_second.append(cut_columns[j])

#Guardar resultados
normalized_spectral_data_df_expanded_filtered.to_excel("Normalized_Spectra.xlsx", index=True)
print ("Normalized_Spectra.xlsx")

# Plotting all baseline-corrected and normalized spectra
num_spectra = len(normalized_spectral_data_df_expanded_filtered)

plt.figure(figsize=(10, 6))
for i in range(num_spectra):
    plotList = normalized_spectral_data_df_expanded_filtered.iloc[i].tolist()
    plt.plot(cut_columns_second, plotList[0:len(cut_columns_second)])
    plt.plot(cut_columns_first, plotList[len(cut_columns_second):])

plt.title('All Spectra (Baseline Corrected, Normalized, and Cut between 1850-2500) - URINE')
plt.xlabel('Wavelength')
plt.ylabel('Intensity')
plt.legend()
plt.xlim((cut_columns.min(), cut_columns.max()))  # Set x-axis limits to include all data
plt.show()

print(f"Number of spectra plotted: {num_spectra}")

# Define os grupos por sexo
group_colors = {1: "blue", 0: "red"}
group_labels = {1: "Female", 0: "Male"}

# Carregar dados espectrais e grupos usando merge
df_spectra = pd.read_excel("Normalized_Spectra.xlsx")
df_groups  = pd.read_excel("Samples group.xlsx")
df = pd.merge(df_spectra, df_groups, left_on=df_spectra.columns[0],
              right_on=df_groups.columns[0], how='left')

# Mapear os grupos
group_map = {"Feminino": 1, "Masculino": 0}
df['Sex'] = df['Sex'].map(group_map).fillna(2).astype(int)

sample_names = df.iloc[:, 0]
spectra = df.iloc[:, 1:df_spectra.shape[1]]
  
print (spectra)

label_bool = {1:0, 0:0, 2:0}

# Plot
plt.figure(figsize=(10, 6))
for i in range(len(df)):
    plotList = spectra.iloc[i].tolist()
    label = None
    if label_bool[df['Sex'].iloc[i]] == 0:
        label = group_labels[df['Sex'].iloc[i]]
        label_bool[df['Sex'].iloc[i]] = 1
    plt.plot(cut_columns_second, plotList[0:len(cut_columns_second)], color=group_colors[df['Sex'].iloc[i]], label=label)
    plt.plot(cut_columns_first, plotList[len(cut_columns_second):], color=group_colors[df['Sex'].iloc[i]])
plt.title('All Spectra by gender')
plt.xlabel('Wavenumber')
plt.ylabel('Intensity')
plt.legend()
plt.tight_layout()
plt.show()

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

label_bool = {1:0, 0:0, 2:0}

# Plot
plt.figure(figsize=(10, 6))
for i in range(len(df)):
    plotList = spectra.iloc[i].tolist()
    label = None
    if label_bool[df['Age'].iloc[i]] == 0:
        label = group_labels[df['Age'].iloc[i]]
        label_bool[df['Age'].iloc[i]] = 1
    plt.plot(cut_columns_second, plotList[0:len(cut_columns_second)], color=group_colors[df['Age'].iloc[i]], label=label)
    plt.plot(cut_columns_first, plotList[len(cut_columns_second):], color=group_colors[df['Age'].iloc[i]])
plt.title('All Spectra by age')
plt.xlabel('Wavenumber')
plt.ylabel('Intensity')
plt.legend()
plt.tight_layout()
plt.show()




