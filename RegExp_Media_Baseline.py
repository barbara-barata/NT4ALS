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

# Salvar os resultados corrigidos
baseline_corrected_spectral_data_df.to_excel("Baseline_Corrected_Spectra.xlsx", index=True)

print("Correção de baseline concluída e salva em 'Baseline_Corrected_Spectra.xlsx'.")


def euclidean_normalization(spectrum):
    euclidean_length = np.linalg.norm(spectrum)
    return spectrum / euclidean_length

# Apply Euclidean normalization to each spectrum
normalized_spectral_data = np.apply_along_axis(euclidean_normalization, axis=1, arr=baseline_corrected_spectral_data)

# Cut intensities between wavenumbers 1850 and 2300
cut_wavelengths_indices = (wavelengths >= 1850) & (wavelengths <= 2500)
normalized_spectral_data[:, cut_wavelengths_indices] = 0

# Plotting all baseline-corrected and normalized spectra
num_spectra = len(normalized_spectral_data)

plt.figure(figsize=(10, 6))
for i in range(num_spectra):
    plt.plot(wavelengths, normalized_spectral_data[i])

plt.title('All Spectra (Baseline Corrected, Normalized, and Cut between 1850-2500) - URINA')
plt.xlabel('Wavelength')
plt.ylabel('Intensity')
plt.legend()
plt.xlim((wavelengths.min(), wavelengths.max()))  # Set x-axis limits to include all data
plt.show()

print(f"Number of spectra plotted: {num_spectra}")





