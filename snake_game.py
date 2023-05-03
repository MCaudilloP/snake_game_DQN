import pygame
import numpy as np

# Configuración del juego
WIDTH, HEIGHT = 400, 400
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLOCK_SIZE = 20

class SnakeGame:
    def __init__(self):
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()

        self.direction = (1, 0)
        self.snake = [(WIDTH // 2, HEIGHT // 2)]
        self.food = self.spawn_food()
        self.grid_size = WIDTH // BLOCK_SIZE  # <-- Add this line


    def spawn_food(self):
        while True:
            food = (np.random.randint(0, WIDTH // BLOCK_SIZE) * BLOCK_SIZE,
                    np.random.randint(0, HEIGHT // BLOCK_SIZE) * BLOCK_SIZE)
            if food not in self.snake:
                return food

    def draw_grid(self):
        for x in range(0, WIDTH, BLOCK_SIZE):
            pygame.draw.line(self.window, WHITE, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, BLOCK_SIZE):
            pygame.draw.line(self.window, WHITE, (0, y), (WIDTH, y))

    def draw_snake(self):
        for segment in self.snake:
            pygame.draw.rect(self.window, GREEN, pygame.Rect(segment[0], segment[1], BLOCK_SIZE, BLOCK_SIZE))

    def draw_food(self):
        pygame.draw.rect(self.window, WHITE, pygame.Rect(self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))

    def move(self):
        head = self.snake[0]
        new_head = (head[0] + self.direction[0] * BLOCK_SIZE, head[1] + self.direction[1] * BLOCK_SIZE)
        self.snake.insert(0, new_head)

        if self.snake[0] == self.food:
            self.food = self.spawn_food()
        else:
            self.snake.pop()

    def check_collision(self):
        head = self.snake[0]
        if head[0] < 0 or head[0] >= WIDTH or head[1] < 0 or head[1] >= HEIGHT:
            return True
        if head in self.snake[1:]:
            return True
        return False

    def play(self):
        game_over = False
        while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_over = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.direction = (0, -1)
                    elif event.key == pygame.K_DOWN:
                        self.direction = (0, 1)
                    elif event.key == pygame.K_LEFT:
                        self.direction = (-1, 0)
                    elif event.key == pygame.K_RIGHT:
                        self.direction = (1, 0)

            self.window.fill(BLACK)
            self.draw_grid()
            self.draw_snake()
            self.draw_food()
            self.move()

            if self.check_collision():
                game_over = True

            pygame.display.flip()
            self.clock.tick(10)

        pygame.quit()

    def get_state(self):
        state = np.zeros((3, self.grid_size, self.grid_size))
        for i, pos in enumerate(self.snake):
            y, x = pos
            if x >= self.grid_size or y >= self.grid_size:
                continue # Skip if position is outside grid boundaries
            if i == 0:
                state[0, y, x] = 1  # Cabeza de la serpiente
            else:
                state[1, y, x] = 1  # Cuerpo de la serpiente
        y, x = self.food
        if x >= self.grid_size or y >= self.grid_size:
            return state # Return the state as is if food is outside grid boundaries
        state[2, y, x] = 1  # Comida
        return state

