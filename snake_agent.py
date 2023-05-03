import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
Input,
Reshape,
Conv2D,
Flatten,
Dense,
)
from tensorflow.keras.optimizers import Adam
import cv2

class SnakeAgent:
    def __init__(self, input_dim, output_dim, grid_size=5):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.build_model()

    def build_model(self):
        inputs = Input(shape=(self.grid_size, self.grid_size, 3))
        x = Conv2D(24, kernel_size=3, activation='relu')(inputs)
        x = Flatten()(x)
        x = Dense(24, activation='relu')(x)
        outputs = Dense(self.output_dim, activation='softmax')(x)
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    def train(self, x_train, y_train, epochs, batch_size):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, x):
        return self.model.predict(x)

    def get_action(self, state):
        state = self.preprocess_state(state)
        action_probs = self.model.predict(state)[0]
        action = np.argmax(action_probs)
        return action


    def q_learning_train(self, states, actions, rewards):
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)

        actions_onehot = np.zeros((len(actions), 4))
        actions_onehot[np.arange(len(actions)), actions] = 1

        self.model.fit(states, actions_onehot, sample_weight=rewards, epochs=1, verbose=0)

    def get_state(self):
        state = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        state[self.snake_head[0], self.snake_head[1], 0] = 1
        for (x, y) in self.snake_body:
            state[x, y, 1] = 1
        state[self.food_position[0], self.food_position[1], 2] = 1
        return state

    def preprocess_state(self, state):
        state = cv2.resize(state, (self.grid_size, self.grid_size), interpolation=cv2.INTER_AREA)  # Actualizar el tamaño del estado
        state = state / 255.0  # Normalizar los valores de píxeles en el rango [0, 1]
        if state.shape != (self.grid_size, self.grid_size, 3):
            state = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        return np.reshape(state, (1, self.grid_size, self.grid_size, 3))

