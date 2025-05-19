import pygame
import numpy as np
import random
import time

# Configurações
GRID_SIZE = 15
CELL_SIZE = 60
WIDTH = HEIGHT = GRID_SIZE * CELL_SIZE
STARTORANGE = (3, 0)
STARTDEEPPINK = (10, 0)
TRAPS = [(0, 8), (7, 0), (4, 9), (14, 14), (14, 1), (1, 2), (8,6), (6,6),(13,8)]
GOAL = (7, 14)

OBSTACLES = [
    # limites horizontais
    (0,7), (14,7), (1,7), (2,7), (4,7), (5,7), (8,7), (9,7), (10,7), (12,7), (13,7), (6,7),
    
    # limites verticais
    (7,13), (7,1), (7,7), (7,2), (7,6), (7,5), (7,3), (7,8), (7,9), (7,11), (7,12),

    # Quadrado superior da esquerda
    (5, 0),
    (2,1), (1,1), (4,1), (5,1),
    (2,2), 
    (2,3), (1,3), (4,3), (5,3),
    (4,4),
    (4,5), (3,5), (2,5), (1,5), (6,5),
    (4,6),

    # Quadrado inferior da esquerda
    (0,14),
    (1,12), (1,10), (1,14), (1,8), (1,9), 
    (2,12), (2,9), (2,14), 
    (3,12),
    (4,11),(4,12), (4,13), (4, 10),
    (5, 11), (5, 13),
    (6, 9),

    # Quadrado superior da direita
    (8, 5),
    (9, 1), (9, 3),(9, 2), 
    (9, 0), (10, 1), (10, 3), (10, 5),
    (12, 1), (12, 4), (12, 5),
    (13, 1), (13, 5),
    (13, 2), (13, 4),
    (14, 2), 

    # Quadrado inferior da direita
    (12, 13), (9, 10), 
    (8, 12), 
    (9, 9), (13, 13),
    (9, 12), (10, 12), (13, 12),
    (14, 13), (10, 9), (8, 13),
    (12, 8), (12, 10), (10, 13),
    (13, 10), (13, 11),
    (14, 8), 
]

# Função para gerar posições válidas para teleportes
def generate_valid_positions(count):
    positions = []
    while len(positions) < count:
        x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
        pos = (x, y)
        if (pos not in OBSTACLES and pos != STARTDEEPPINK and pos != STARTORANGE and pos != GOAL and 
            pos not in TRAPS and pos not in positions):
            positions.append(pos)
    return positions

# Gerar pares de teleportes aleatoriamente
NUM_TELEPORT_PAIRS = 3
valid_positions = generate_valid_positions(NUM_TELEPORT_PAIRS * 2)
TELEPORT_PAIRS = [(valid_positions[i], valid_positions[i+1]) for i in range(0, len(valid_positions), 2)]

print("Teleportes gerados:", TELEPORT_PAIRS)

ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # cima, baixo, esquerda, direita

# Parâmetros do Q-Learning
ALPHA = 0.2    
GAMMA = 0.95     
EPSILON = 1.0     
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.998  
EPISODES = 1000   
MAX_STEPS = 200    

# Cores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)
BLUE = (50, 50, 255)
DEEPPINK = (255,20,147)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
ORANGE = (255, 165, 0)
GOLD = (255, 215, 0)
BROWN = (165, 42, 42)
PURPLE = (128, 0, 128)
CYAN = (0, 255, 255)

# Q-table
q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))

# Funções auxiliares
def is_valid(pos):
    x, y = pos
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and pos not in OBSTACLES

def get_next_state(state, action):
    dx, dy = ACTIONS[action]
    next_state = (state[0] + dx, state[1] + dy)
    return next_state if is_valid(next_state) else state

def get_teleport_target(pos):
    for entrance, exit in TELEPORT_PAIRS:
        if pos == entrance:
            return exit
        if pos == exit:
            return entrance
    return None

def get_reward(state, steps_taken):
    if state == GOAL:
        return 1000 - steps_taken
    elif get_teleport_target(state) is not None:
        return -5
    elif state in TRAPS:
        return -500
    elif state in OBSTACLES:
        return -100
    else:
        return -1

def draw_grid(screen):
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = WHITE
            if (x, y) in OBSTACLES:
                color = BLACK
            elif (x, y) == STARTORANGE:
                color = GREEN
            elif (x, y) == STARTDEEPPINK:
                color = GREEN
            elif (x, y) == GOAL:
                color = RED
            elif any((x, y) in pair for pair in TELEPORT_PAIRS):
                color = CYAN
            elif (x, y) in TRAPS:
                color = BROWN
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, GREY, rect, 1)

            if (x, y) in TRAPS:
                pygame.draw.line(screen, RED, 
                               (x * CELL_SIZE + 10, y * CELL_SIZE + 10),
                               (x * CELL_SIZE + CELL_SIZE - 10, y * CELL_SIZE + CELL_SIZE - 10), 3)
                pygame.draw.line(screen, RED, 
                               (x * CELL_SIZE + CELL_SIZE - 10, y * CELL_SIZE + 10),
                               (x * CELL_SIZE + 10, y * CELL_SIZE + CELL_SIZE - 10), 3)
            elif any((x, y) in pair for pair in TELEPORT_PAIRS):
                font = pygame.font.SysFont(None, 30)
                text = font.render("T", True, BLACK)
                screen.blit(text, (x * CELL_SIZE + CELL_SIZE//2 - 5, y * CELL_SIZE + CELL_SIZE//2 - 10))

def draw_agent(screen, pos, color=BLUE):
    rect = pygame.Rect(pos[0] * CELL_SIZE + 10, pos[1] * CELL_SIZE + 10, CELL_SIZE - 20, CELL_SIZE - 20)
    pygame.draw.ellipse(screen, color, rect)

def process_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

# -----------------------------
# TREINAMENTO
# -----------------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Treinamento do Agente")
clock = pygame.time.Clock()

for episode in range(EPISODES):
    # Escolhe aleatoriamente qual agente treinar neste episódio
    state = STARTORANGE if random.random() < 0.5 else STARTDEEPPINK
    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
    
    for step in range(MAX_STEPS):
        process_events()

        if random.random() < EPSILON:
            action = random.randint(0, 3)
        else:
            action = np.argmax(q_table[state[0], state[1]])
        
        next_state = get_next_state(state, action)
        
        teleport_target = get_teleport_target(next_state)
        if teleport_target is not None:
            next_state = teleport_target
        
        reward = get_reward(next_state, step)
        
        old_value = q_table[state[0], state[1], action]
        next_max = np.max(q_table[next_state[0], next_state[1]])
        new_value = old_value + ALPHA * (reward + GAMMA * next_max - old_value)
        q_table[state[0], state[1], action] = new_value
        
        state = next_state
        
        if episode % 50 == 0:
            screen.fill(WHITE)
            draw_grid(screen)
            draw_agent(screen, state, ORANGE if state == STARTORANGE else DEEPPINK)
            pygame.display.flip()
            clock.tick(20)
        
        if state == GOAL:
            break

    if episode % 100 == 0:
        print(f"Episódio {episode}, Epsilon: {EPSILON:.3f}")

print("Treinamento concluído!")
pygame.quit()

# -----------------------------
# COMPARAÇÃO DE CAMINHOS
# -----------------------------
def find_shortest_path(start_pos):
    path = [start_pos]
    current_pos = start_pos
    visited = set()
    visited.add(current_pos)
    steps = 0
    teleport_used = False
    
    while current_pos != GOAL and steps < MAX_STEPS:
        action = np.argmax(q_table[current_pos[0], current_pos[1]])
        next_pos = get_next_state(current_pos, action)
        
        teleport_target = get_teleport_target(next_pos)
        if teleport_target is not None:
            next_pos = teleport_target
            teleport_used = True
        
        if next_pos in visited:
            break
        visited.add(next_pos)
        path.append(next_pos)
        current_pos = next_pos
        steps += 1
    
    return path, teleport_used

path_orange, teleport_orange = find_shortest_path(STARTORANGE)
path_deeppink, teleport_deeppink = find_shortest_path(STARTDEEPPINK)

print("\nComparação de caminhos:")
print(f"Agente ORANGE: {len(path_orange)} passos, Teleporte usado: {'Sim' if teleport_orange else 'Não'}")
print(f"Agente DEEPPINK: {len(path_deeppink)} passos, Teleporte usado: {'Sim' if teleport_deeppink else 'Não'}")

if len(path_orange) < len(path_deeppink):
    print("Agente ORANGE teve o caminho mais curto!")
    best_path = path_orange
    best_color = ORANGE
elif len(path_deeppink) < len(path_orange):
    print("Agente DEEPPINK teve o caminho mais curto!")
    best_path = path_deeppink
    best_color = DEEPPINK
else:
    print("Ambos os agentes tiveram caminhos de mesmo comprimento!")
    best_path = path_orange
    best_color = PURPLE

# Visualização do melhor caminho
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Caminho Mais Curto")
clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    screen.fill(WHITE)
    draw_grid(screen)
    
    for i, pos in enumerate(best_path):
        progress = i / len(best_path)
        r = int(best_color[0] * (1 - progress) + 255 * progress)
        g = int(best_color[1] * (1 - progress) + 255 * progress)
        b = int(best_color[2] * (1 - progress))
        color = (r, g, b)
        draw_agent(screen, pos, color)
    
    draw_agent(screen, best_path[0], GREEN)
    draw_agent(screen, GOAL, RED)
    
    font = pygame.font.SysFont(None, 36)
    text = font.render(f'Melhor caminho: {len(best_path)} passos', True, BLACK)
    screen.blit(text, (10, 10))
    
    pygame.display.flip()
    clock.tick(30)

pygame.quit()