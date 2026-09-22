import random
import numpy as np
import pygame
import matplotlib.pyplot as plt

GRID_SIZE = 20
TILE_SIZE = 20
WIDTH, HEIGHT = GRID_SIZE * TILE_SIZE, GRID_SIZE * TILE_SIZE
DIRECTIONS =[(-1,0), (0,1), (1,0), (0,-1)]  # Up, Right, Down, Left

class SnakeGame:
    """Manages snakes rules, grid pyshics and the state of the game"""

    def __init__(self, grid_size=GRID_SIZE):
        self.grid_size = grid_size
        self.reset()
        
    def reset(self):
        """Resets the game state to the initial configuration"""
        self.snake = [(10,10), (10,11)]
        self.dir = 0
        self.score = 0
        self.food = self._place_food()
        self.is_over = False
        return self
    
    def _place_food(self):
        while True:
            food = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            if food not in self.snake:
                return food
            
    def step(self, action):
        if self.is_over:
            return False, True

        if action == 1:
            self.dir = (self.dir + 1) % 4
        elif action == 2:
            self.dir = (self.dir - 1) % 4
        
        head = (self.snake[0][0] + DIRECTIONS[self.dir][0], self.snake[0][1] + DIRECTIONS[self.dir][1])
        
        if (head in self.snake) or not (0 <= head[0] < self.grid_size and 0 <= head[1] < self.grid_size):
            self.is_over = True
            return False, True
        
        ate_food = (head == self.food)
        if ate_food:
            self.score += 1
            self.snake.insert(0, head)
            self.food = self._place_food()  
        else:
            self.snake.insert(0, head)
            self.snake.pop()
        return ate_food, False
    
class QLearningAgent:
    """Handles Q-Table state updates and action selection"""

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, min_epsilon=0.01, epsilon_decay=0.995):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

    def _get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]
        return self.q_table[state]

    def choose_action(self, state):
        # Clean epsilon-greedy search
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        q_vals = self._get_q_values(state)
        return int(np.argmax(q_vals))

    def update(self, state, action, reward, next_state, done):
        q_vals = self._get_q_values(state)
        
        if done:
            target = reward
        else:
            next_q_vals = self._get_q_values(next_state)
            target = reward + self.gamma * max(next_q_vals)

        q_vals[action] += self.alpha * (target - q_vals[action])

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

class SnakeEnvironment:

    def __init__(self, game: SnakeGame):
        self.game = game

    def get_state(self):
        head_r, head_c = self.game.snake[0]
        dir_idx = self.game.dir
        left_dir = (dir_idx - 1) % 4
        right_dir = (dir_idx + 1) % 4

        def is_danger(d):
            dr, dc = DIRECTIONS[d]
            r, c = head_r + dr, head_c + dc
            if (r < 0 or r >= self.game.grid_size or 
                c < 0 or c >= self.game.grid_size or 
                (r, c) in self.game.snake[:-1]):
                return 1
            return 0

        # Vector from head to food in (row, col) coordinates
        food_r, food_c = self.game.food
        df_r = food_r - head_r
        df_c = food_c - head_c

        # Forward vector
        dr, dc = DIRECTIONS[dir_idx]
        
        # Left vector
        ldr, ldc = DIRECTIONS[left_dir]
        
        # Right vector
        rdr, rdc = DIRECTIONS[right_dir]

        # Dot products for relative direction
        food_ahead = 1 if (df_r * dr + df_c * dc) > 0 else 0
        food_left  = 1 if (df_r * ldr + df_c * ldc) > 0 else 0
        food_right = 1 if (df_r * rdr + df_c * rdc) > 0 else 0

        return (
            is_danger(dir_idx),
            is_danger(left_dir),
            is_danger(right_dir),
            food_ahead,
            food_left,
            food_right
        )

    def step(self, action):
        state = self.get_state()
        
        # Track distance before move to reward getting closer
        head = self.game.snake[0]
        food = self.game.food
        old_dist = abs(head[0] - food[0]) + abs(head[1] - food[1])

        ate_food, game_over = self.game.step(action)

        new_head = self.game.snake[0]
        new_dist = abs(new_head[0] - food[0]) + abs(new_head[1] - food[1])

        if game_over:
            reward = -10
        elif ate_food:
            reward = 10
        elif new_dist < old_dist:
            reward = 0.1   # Small positive reward for stepping toward food
        else:
            reward = -0.2  # Small penalty for stepping away from food

        next_state = self.get_state()
        return state, action, reward, next_state, game_over
    
class ScorePlotter:
    """Renders a live chart of scores and running average."""

    def __init__(self):
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.scores = []
        self.mean_scores = []
        
        self.ax.set_title("Q-Learning Snake agent progress")
        self.ax.set_xlabel("Attempt")
        self.ax.set_ylabel("Score")
        
        self.line_score, = self.ax.plot([], [], label="Score", alpha=0.5, color="skyblue")
        self.line_mean, = self.ax.plot([], [], label="10-Game Mean", color="darkblue", linewidth=2)
        self.ax.legend(loc="upper left")
        self.fig.tight_layout()

    def add_score(self, score: int):
        self.scores.append(score)
        
        # Calculate moving average of last 10 games
        recent_scores = self.scores[-10:]
        self.mean_scores.append(np.mean(recent_scores))

        attempts = list(range(1, len(self.scores) + 1))

        self.line_score.set_data(attempts, self.scores)
        self.line_mean.set_data(attempts, self.mean_scores)

        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
class PygameRenderer:
    """Handles rendering to the Pygame screen independently."""

    def __init__(self, tile_size=TILE_SIZE, grid_size=GRID_SIZE):
        self.tile_size = tile_size
        width, height = grid_size * tile_size, grid_size * tile_size
        pygame.init()
        self.font = pygame.font.SysFont("Arial", 18)
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Snake AI - Simplified SOLID")
        self.clock = pygame.time.Clock()

    def render(self, game: SnakeGame, attempts: int, epsilon: float):
        self.screen.fill((20, 20, 20))
        
        # Draw Food
        pygame.draw.rect(
            self.screen,
            (230, 50, 50),
            (game.food[1] * self.tile_size, game.food[0] * self.tile_size, self.tile_size - 2, self.tile_size - 2),
        )
        
        # Draw Snake
        for r, c in game.snake:
            pygame.draw.rect(
                self.screen,
                (50, 230, 100),
                (c * self.tile_size, r * self.tile_size, self.tile_size - 2, self.tile_size - 2),
            )


        text_surface = self.font.render(
            f"Attempt: {attempts} | Score: {game.score} | Epsilon: {epsilon:.2f}", 
            True, 
            (255, 255, 255)
        )
        self.screen.blit(text_surface, (10, 10))

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    game = SnakeGame()
    env = SnakeEnvironment(game)
    agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=1.0, min_epsilon=0.01, epsilon_decay=0.985)
    renderer = PygameRenderer()
    plotter = ScorePlotter()

    attempts = 1
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        state = env.get_state()
        action = agent.choose_action(state)

        old_state, action, reward, next_state, game_over = env.step(action)
        agent.update(old_state, action, reward, next_state, game_over)

        if game_over:
            print(f"Game Over! Attempt: {attempts} | Score: {game.score}")
            plotter.add_score(game.score)
            attempts += 1
            agent.decay_epsilon()
            
            if attempts >= 750:
                running = False
            else:
                game.reset()

        renderer.render(game, attempts, agent.epsilon)

    renderer.close()
    plt.close(plotter.fig)  # Close the figure after the game loop ends