import pygame
import numpy as np
import random
import time

# Configurações
GRID_SIZE = 10
CELL_SIZE = 60
WIDTH = HEIGHT = GRID_SIZE * CELL_SIZE
START = (0, 0)
TRAPS = [(0, 9), (9, 0), (4, 9)]
TREASURE = [(8, 8), (2, 6)]
GOAL = (9, 9)
OBSTACLES = [(0,1), (1,1), (2,1), (4,1), (5,1), (7,1), (8,1), (9,1),
            (2,3), (1,3), (2,2), (0,5), (1,5), (2,5), (0,7), (0,6),
            (1,7), (1,8), (3,5), (4,3), (5,3), (6,3), (4,4), (6,5),
            (5,6), (4,7), (3,8), (3,8), (3,9), (8,6), (8,2), (9,6),
            (5,8), (4,8), (7,8), (5,0), (8,7), (7,7), (8,3),(8,4)]

ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # cima, baixo, esquerda, direita

# Parâmetros do Q-Learning
ALPHA = 0.2       # Taxa de aprendizado aumentada
GAMMA = 0.95      # Fator de desconto
EPSILON = 1.0     # Exploração inicial máxima
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.998  # Decaimento mais lento
EPISODES = 1000   # Mais episódios de treinamento
MAX_STEPS = 200    # Passos máximos por episódio

# Cores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)
BLUE = (50, 50, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
ORANGE = (255, 165, 0)
GOLD = (255, 215, 0)
BROWN = (165, 42, 42)
PURPLE = (128, 0, 128)

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

def get_reward(state, steps_taken):
    """Sistema de recompensas balanceado"""
    if state == GOAL:
        return 1000 - steps_taken  # Recompensa maior e penalidade por demora
    elif state in TREASURE:
        return 100 - (steps_taken * 0.5)  # Tesouros valem menos que objetivo
    elif state in TRAPS:
        return -500  # Penalidade alta por armadilhas
    elif state in OBSTACLES:
        return -100
    else:
        return -1  # Penalidade padrão por movimento

def draw_grid(screen):
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = WHITE
            if (x, y) in OBSTACLES:
                color = BLACK
            elif (x, y) == START:
                color = GREEN
            elif (x, y) == GOAL:
                color = RED
            elif (x, y) in TREASURE:
                color = GOLD
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
            elif (x, y) in TREASURE:
                pygame.draw.circle(screen, PURPLE, 
                                 (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2), 
                                 CELL_SIZE // 3)

def draw_agent(screen, pos, color=BLUE):
    rect = pygame.Rect(pos[0] * CELL_SIZE + 10, pos[1] * CELL_SIZE + 10, CELL_SIZE - 20, CELL_SIZE - 20)
    pygame.draw.ellipse(screen, color, rect)

def process_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

# -----------------------------
# TREINAMENTO COM VISUALIZAÇÃO
# -----------------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Treinamento do Agente")
clock = pygame.time.Clock()

for episode in range(EPISODES):
    state = START
    collected_treasures = []
    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
    
    for step in range(MAX_STEPS):
        process_events()

        # Seleção de ação ε-greedy
        if random.random() < EPSILON:
            action = random.randint(0, 3)  # Exploração
        else:
            action = np.argmax(q_table[state[0], state[1]])  # Exploração
        
        next_state = get_next_state(state, action)
        reward = get_reward(next_state, step)
        
        # Atualização Q-learning
        old_value = q_table[state[0], state[1], action]
        next_max = np.max(q_table[next_state[0], next_state[1]])
        new_value = old_value + ALPHA * (reward + GAMMA * next_max - old_value)
        q_table[state[0], state[1], action] = new_value
        
        state = next_state
        
        # Visualização apenas a cada 50 episódios para melhor performance
        if episode % 50 == 0:
            screen.fill(WHITE)
            draw_grid(screen)
            draw_agent(screen, state, ORANGE)
            pygame.display.flip()
            clock.tick(500)
        
        if state == GOAL:
            break

    if episode % 100 == 0:
        print(f"Episódio {episode}, Epsilon: {EPSILON:.3f}")

print("Treinamento concluído!")
time.sleep(1)
pygame.display.quit()

# -----------------------------
# EXECUÇÃO DO AGENTE TREINADO
# -----------------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Execução do Agente Treinado")
clock = pygame.time.Clock()

agent_pos = START
path = [agent_pos]
collected_treasures = []
reached_goal = False
running = True
score = 0

while running:
    process_events()

    screen.fill(WHITE)
    draw_grid(screen)

    # Desenha caminho percorrido
    for pos in path:
        draw_agent(screen, pos, BLUE)

    # Desenha tesouros coletados
    for i, treasure in enumerate(collected_treasures):
        x, y = treasure
        pygame.draw.circle(screen, PURPLE, (20 + i * 40, HEIGHT - 20), 15)

    # Desenha agente atual
    if not reached_goal:
        draw_agent(screen, agent_pos, BLUE)

    # Mostra informações
    font = pygame.font.SysFont(None, 36)
    score_text = font.render(f'Score: {score}', True, BLACK)
    screen.blit(score_text, (10, 10))
    
    # Mostra direção preferencial
    best_action = np.argmax(q_table[agent_pos[0], agent_pos[1]])
    action_arrows = ["↑", "↓", "←", "→"]
    arrow_text = font.render(f"Direção: {action_arrows[best_action]}", True, BLUE)
    screen.blit(arrow_text, (WIDTH//2 - 50, 10))

    pygame.display.flip()
    clock.tick(5)  # Velocidade mais lenta para acompanhar

    if not reached_goal:
        if agent_pos != GOAL:
            action = np.argmax(q_table[agent_pos[0], agent_pos[1]])
            next_pos = get_next_state(agent_pos, action)
            
            if next_pos == agent_pos:
                print("Agente está preso!")
                reached_goal = True
            else:
                path.append(next_pos)
                agent_pos = next_pos
                
                # Verifica tesouros
                if agent_pos in TREASURE and agent_pos not in collected_treasures:
                    collected_treasures.append(agent_pos)
                    score += 100
                    print(f"Tesouro coletado em {agent_pos}! Pontuação: {score}")
                
                # Verifica armadilhas
                if agent_pos in TRAPS:
                    score -= 50
                    print(f"Armadilha em {agent_pos}! Pontuação: {score}")
                
                time.sleep(0.5)
        else:
            reached_goal = True
            score += 1000
            print(f"\nMissão completa! Caminho: {path}")
            print(f"Tesouros coletados: {len(collected_treasures)}")
            print(f"Pontuação final: {score}")

pygame.quit()