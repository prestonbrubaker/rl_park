import pygame
import random
import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
import pickle

pygame.init()

WIDTH, HEIGHT = 600, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

AI_CONTROL = True
RECORD = False
APPEND_DATA = False

class WillohNet(nn.Module):
    def __init__(self):
        super(WillohNet, self).__init__()
        self.fc1 = nn.Linear(1203, 1203)
        self.fc2 = nn.Linear(1203, 300)
        self.fc3 = nn.Linear(300, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 1)

        self.fc6 = nn.Linear(1203,1)

    def forward(self, x):
        y = x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x) + self.fc6(y)
        return x
    

willohnet = WillohNet()
willohnet.load_state_dict(torch.load('willohnet.pth'))
willohnet.eval()

font = pygame.font.SysFont("roboto", 24)

dt = 1 / 60
g = 200
player_speed = 300
jump_strength = -300

platforms = []
furthest_y = 100
platform_w = 20
platform_h = 20

ground_height = 100
camera_y = -(HEIGHT - ground_height)

player_size = 25
player_pos = [WIDTH // 2, -player_size]
player_vel_y = 0

score = 0
max_alt = 0
plat_pen = 0
on_ground = True

data = []
labels = []

def reset_game():
    global platforms
    global score
    global max_alt
    global camera_y
    global player_pos
    global furthest_y
    global plat_pen
    platforms = []
    score = 0
    max_alt = 0
    camera_y = -(HEIGHT - ground_height)
    player_pos = [WIDTH // 2, -player_size]
    furthest_y = 100
    plat_pen = 0

    generate_platforms(camera_y + HEIGHT)



def handle_camera():
    global camera_y
    target_camera_y = player_pos[1] - HEIGHT // 2
    camera_y += (target_camera_y - camera_y) * 0.3

def detect_collision(player, platform):
    return (
        player[0] < platform[0] + platform[2]
        and player[0] + player[2] > platform[0]
        and player[1] < platform[1] + platform[3]
        and player[1] + player[3] > platform[1]
    )

def check_on_surface():
    global on_ground, player_vel_y
    player_rect = [player_pos[0], player_pos[1], player_size, player_size]
    lowest_y = None
    for platform in platforms:
        platform_rect = [platform["x"], platform["y"], platform["width"], platform["height"]]
        if detect_collision(player_rect, platform_rect) and player_vel_y >= 0:
            if lowest_y is None or platform["y"] < lowest_y:
                lowest_y = platform["y"]
    
    if lowest_y is None and player_pos[1] + player_size >= 0:
        lowest_y = 0
    
    return lowest_y

def physics():
    global player_vel_y, player_pos, max_alt, score, plat_pen, on_ground
    player_vel_y += g * dt
    player_vel_y = max(min(player_vel_y, 500), -500)
    new_y = check_on_surface()
    if new_y is not None and player_pos[1] + player_size + player_vel_y * dt >= new_y:
        player_pos[1] = new_y - player_size
        player_vel_y = 0
        on_ground = True
        score += 0.1
    else:
        player_pos[1] += player_vel_y * dt
        on_ground = False
    world_y = -player_pos[1]
    if world_y > max_alt:
        max_alt = world_y
        #score = round((max_alt - plat_pen * 100) / 100, 2)
    score = HEIGHT + 300 + ground_height - player_pos[1] + screen_y

def generate_platforms(max_alt):
    global furthest_y
    while furthest_y > max_alt - HEIGHT:
        platform_width = 50
        platform_height = 20
        platform_x = random.randint(0, WIDTH - platform_width)
        platform_y = furthest_y - random.randint(100, 180)  # Increased vertical spacing
        platforms.append({
            "x": platform_x,
            "y": platform_y, 
            "width": platform_width,
            "height": platform_height
        })
        furthest_y = platform_y
        if len(platforms) > 100:
            break

def update_platforms():
    for platform in platforms:
        # Scale shake amplitude: minimal at y=0, increasing as y becomes more negative
        amplitude = 0.5 - platform["y"] / 2000  # 0.5 at y=0, ~1.5 at y=-2000
        dx = random.uniform(-amplitude, amplitude) * dt * (WIDTH - platform["width"])
        platform["x"] += dx
        if platform["x"] < 0:
            platform["x"] = 0
        if platform["x"] > WIDTH - platform["width"]:
            platform["x"] = WIDTH - platform["width"]

generate_platforms(camera_y + HEIGHT)

itC = 0

outs = [0] * 3

running = True
clock = pygame.time.Clock()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    

    old_score = score

    
    if camera_y > furthest_y:
        generate_platforms(camera_y)

    platforms = [platform for platform in platforms if platform["y"] + platform["height"] > camera_y - HEIGHT * 2]
    
    screen.fill((50, 50, 50))
    pygame.draw.rect(screen, (100, 100, 100), (0, -camera_y, WIDTH, HEIGHT // 2))
    for platform in platforms:
        screen_y = platform["y"] - camera_y
        pygame.draw.rect(screen, WHITE, (platform["x"], screen_y, platform["width"], platform["height"]))
    pygame.draw.rect(screen, BLACK, (player_pos[0], player_pos[1] - camera_y, player_size, player_size))
    
    position_text = font.render(f'X: {int(player_pos[0])}   Y: {abs(int(player_pos[1]) + 25)}   ON PLAT: {on_ground}   SCORE: {int(score)}', True, WHITE)
    screen.blit(position_text, (10, 10))
    position_text = font.render(f'ItC: {itC}   OUTS (ad[sp]): {outs}', True, WHITE)
    screen.blit(position_text, (10, 30))

    # Capture and process screen
    screen_data = pygame.surfarray.array3d(screen)
    screen_data = screen_data / 255.0
    screen_tensor = torch.tensor(screen_data, dtype=torch.float32)

    print(screen_tensor.shape)

    scale = 20

    cond_screen_data = []
    cond_screen_data_flatlist = []

    for x in range(int(WIDTH / scale)):
        temp_l = []
        for y in range(int(HEIGHT / scale)):
            temp_l.append(screen_data[x * scale][y * scale][0])
            cond_screen_data_flatlist.append(screen_data[x * scale][y * scale][0])
            intensity = screen_data[x * scale][y * scale][0] * 255
            color = (intensity, intensity, intensity)
            pygame.draw.rect(screen, color, (x,y,1,1))
        cond_screen_data.append(temp_l)
    
    cond_screen_data_tensor = torch.tensor(cond_screen_data, dtype=torch.float32).reshape(-1)

    print(cond_screen_data_tensor.shape)

    pygame.display.flip()

    max_o = -1000000
    max_i = 0
    
    for i in range(3):
        control = [0] * 3
        control[i] = 1
        inputt = copy.deepcopy(cond_screen_data_flatlist)
        for j in range(3):
            inputt.append(control[j])
        inputtt = torch.tensor(inputt, dtype=torch.float32)
        output = willohnet(inputtt)
        outs[i] = round(float(output),2)
        if output > max_o:
            max_o = output
            max_i = i
    
    control = [0] * 3
    control[max_i] = 1
    inputt = copy.deepcopy(cond_screen_data_flatlist)
    for j in range(3):
        inputt.append(control[j])
    #inputtt = torch.tensor(inputt, dtype=torch.float32)

    keys = pygame.key.get_pressed()
    if keys[pygame.K_a] or keys[pygame.K_LEFT] or (max_i == 0 and AI_CONTROL):
        player_pos[0] -= player_speed * dt
    if keys[pygame.K_d] or keys[pygame.K_RIGHT] or (max_i == 1 and AI_CONTROL):
        player_pos[0] += player_speed * dt
    if (keys[pygame.K_SPACE] or (max_i == 2 and AI_CONTROL)) and on_ground:
        player_vel_y = jump_strength
        on_ground = False

    player_pos[0] = max(0, min(player_pos[0], WIDTH - player_size))

    physics()
    check_on_surface()
    handle_camera()
    update_platforms()

    ds = score - old_score
    data.append(inputt)
    labels.append([ds])

    if itC % 10000 == 0 and RECORD:
        if APPEND_DATA:
            with open('data.pkl', 'wb') as f:
                data = pickle.load(f)
            with open('labels.pkl', 'wb') as f: 
                labels = pickle.load(f)
        else:
            with open('data.pkl', 'wb') as f:
                pickle.dump(data, f)
            with open('labels.pkl', 'wb') as f:
                pickle.dump(labels, f)
        

    #clock.tick(1)
    itC += 1

    if itC % 3000 == 0:
        reset_game()

pygame.quit()
