import pygame
import random

pygame.init()

font = pygame.font.SysFont("roboto", 24)


window = pygame.display.set_mode((800,800))
height = window.get_height()
width = window.get_width()
print(height)

scroll_y = 0

dt = 0.008
g = 20
WHITE = (255, 255, 255)

platforms = []
platform_w = 100
platform_h = 20

score = 0


class Player:
    def __init__(self, x, y, vel_y):
        self.x = x
        self.y = y
        self.vel_y = vel_y
        self.on_ground = True
        

def check_boundries():
    global scroll_y
    global score
    on_plat = False
    for platform in platforms:
        if player.x > platform[0] and player.x < platform[0] + platform_w and player.y + 50 > platform[1] + scroll_y and player.y + 50 < platform[1] + platform_h + scroll_y and player.vel_y > 0:
            player.on_ground = True
            on_plat =True
            player.vel_y = 0
            player.y = platform[1] - 50 + scroll_y
            if platform[2] == 1:
                score += 1
                platform[2] = 0

    if player.x < 0:
        player.x = 0
    elif player.x >= width - 50:
        player.x = width -50
    if player.y < 0:
        player.y = 0
        scroll_y += player.vel_y


    

    if player.y >= height - 150 + scroll_y:
        player.y = height -150 + scroll_y
        player.on_ground = True
    elif not on_plat:
        player.on_ground = False
    
    


def apply_gravity():
    if not player.on_ground:
        player.vel_y += g * dt

def physic():
    global scroll_y
    player.y += player.vel_y * dt
    if player.y < 0:
        scroll_y -= player.y
        player.y = 0

def create_platform(i):
    platforms.append([
            random.uniform(0, width),
            random.uniform(-200 * i, height),
            1
    ])

def update_platforms():
    for platform in platforms:
        dx = random.uniform(-1,1) * dt * (width - platform[1]) * 0.1
        platform[0] += dx
        if platform[0] < 0:
            platform[0] = 0
        if platform[0] > width:
            platform[0] = width
        


player = Player(x=0,y=650, vel_y=0)

for i in range(1000):
    create_platform(i)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    keys = pygame.key.get_pressed()
    # if keys[pygame.K_w]:
    #     player.y -= 300 * dt
    # if keys[pygame.K_s]:
    #     player.y += 300 * dt
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        player.x -= 100 * dt
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        player.x += 100 * dt
    if keys[pygame.K_SPACE] and player.on_ground:
        player.vel_y -= 100
        player.on_ground = False

    update_platforms()
    apply_gravity()
    physic()
    check_boundries()
    
    
    window.fill((0,0,0))
    pygame.draw.rect(window,(255,0,0),(player.x,player.y,50,50))
    pygame.draw.rect(window, (255, 255, 0), (0,700 + scroll_y,800,100))
    position_text = font.render(f'X: {int(player.x)}   Y: {int(player.y)}   ON PLAT: {player.on_ground}     SCORE: {score}', True, WHITE)
    window.blit(position_text, (10, 10))
    for platform in platforms:
        pygame.draw.rect(window, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), (platform[0], platform[1] + scroll_y,100,10))
    pygame.display.flip()

pygame.quit()