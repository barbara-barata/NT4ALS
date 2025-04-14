import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necessário para gráficos 3D

# Cria a pasta de saída para os plots
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Valores padrão para cores e rótulos de grupos
DEFAULT_GROUP_COLORS = {1: "blue", 0: "red", 2: "gray"}
DEFAULT_GROUP_LABELS = {1: "Patient", 0: "Control", 2: "Unknown"}

# Função para gerar um nome de arquivo baseado no título do plot
def get_filename(title):
    filename = title.replace(" ", "_").replace("(", "").replace(")", "") + ".png"
    return os.path.join(output_dir, filename)

def plot_2d_scatter(df, x, y, title, group_colors=DEFAULT_GROUP_COLORS, group_labels=DEFAULT_GROUP_LABELS, xlabel=None, ylabel=None, legend_kwargs=None):
    plt.figure(figsize=(8,6))
    for group, color in group_colors.items():
        mask = df['Group'] == group
        plt.scatter(df.loc[mask, x], df.loc[mask, y], color=color,
                    label=group_labels.get(group, group), alpha=0.7)
    # Se xlabel ou ylabel não forem passados, utiliza os nomes das colunas
    plt.xlabel(xlabel if xlabel is not None else x)
    plt.ylabel(ylabel if ylabel is not None else y)
    plt.title(title)
    plt.legend(**(legend_kwargs or {}))
    filename = get_filename(title)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plot_3d_scatter(df, x, y, z, title, group_colors=DEFAULT_GROUP_COLORS, group_labels=DEFAULT_GROUP_LABELS, xlabel=None, ylabel=None, zlabel=None, legend_kwargs=None):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    for group, color in group_colors.items():
        mask = df['Group'] == group
        ax.scatter(df.loc[mask, x], df.loc[mask, y], df.loc[mask, z],
                   color=color, label=group_labels.get(group, group), alpha=0.7)
    ax.set_xlabel(xlabel if xlabel is not None else x)
    ax.set_ylabel(ylabel if ylabel is not None else y)
    ax.set_zlabel(zlabel if zlabel is not None else z)
    ax.set_title(title)
    ax.legend(**(legend_kwargs or {}))
    filename = get_filename(title)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()