# snake_plot.py

# Importar las librerías necesarias
import matplotlib.pyplot as plt

# Función para graficar los resultados del entrenamiento
def plot_results(scores, avg_scores):
    plt.plot(scores, label='Puntuaciones')
    plt.plot(avg_scores, label='Puntuaciones promedio')
    plt.xlabel('Episodio')
    plt.ylabel('Puntuación')
    plt.legend()
    plt.show()
