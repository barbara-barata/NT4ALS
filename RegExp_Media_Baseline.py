import pandas as pd
import re
from BaselineRemoval import BaselineRemoval


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
corrected_spectra = data_agrupada.apply(corrigir_baseline, axis=1)

# Converter para DataFrame
corrected_spectra_df = pd.DataFrame(corrected_spectra)

# Salvar os resultados corrigidos
corrected_spectra_df.to_excel("Corrected_Spectra.xlsx", index=True)

print("Correção de baseline concluída e salva em 'Corrected_Spectra.xlsx'.")
