import pygame
import numpy as np
from snake_game import SnakeGame
from snake_agent import SnakeAgent

# Configuración del juego
input_dim = 12  # Dimensiones del estado del juego
output_dim = 4  # Número de acciones posibles (arriba, abajo, izquierda, derecha)
game = SnakeGame()
agent = SnakeAgent(input_dim, output_dim)#input_dim, output_dim

# Bucle de entrenamiento
EPISODES = 100  # Número de episodios de entrenamiento
for episode in range(EPISODES):
    game = SnakeGame()  # Reiniciar el juego para cada episodio
    state = game.get_state()  # Obtener el estado inicial del juego
    done = False
    while not done:
        action = agent.get_action(state)  # Obtener la acción del agente
        next_state, reward, done = game.play_step(action)  # Realizar un paso en el juego
        agent.train(state, action, reward)  # Entrenar al agente con el estado, acción y recompensa
        state = next_state  # Actualizar el estado del juego

# Juego en tiempo real con el agente entrenado
game = SnakeGame()
state = game.get_state()
done = False
while not done:
    action = agent.get_action(state)
    next_state, reward, done = game.play_step(action)
    state = next_state
