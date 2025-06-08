import pygame
import random


pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))

WHITE = (255, 255, 255)
GREY = (50, 50, 50)
LIGHT_GREY = (100, 100, 100)
RED = (255, 0, 0)
PINK = (248, 24, 148)

clock = pygame.time.Clock()
font = pygame.font.SysFont("roboto", 24)

gravity = 1
score = 0
game_over = False
ground_height = 50
camera_x = 0

player_size = 40
player_pos = [WIDTH // 2, HEIGHT - ground_height - player_size]
player_velocity_y = 0
jump_strength = -15
player_speed = 7
on_ground = False

platforms = []
furthest_x = WIDTH

def generate_platforms(max_x):
    """Generate new platforms starting from max_x."""
    global furthest_x
    while furthest_x < max_x + WIDTH * 2:
        platform_width = random.randint(100, 300)
        platform_height = 20
        platform_x = furthest_x + random.randint(100, 250)
        platform_y = random.randint(HEIGHT - ground_height - 200, HEIGHT - ground_height - 50)

        platforms.append({
            "x": platform_x,
            "y": platform_y,
            "width":platform_width,
            "height":platform_height,
            "visited": False
        })
        furthest_x = platform_x + platform_width


def handle_camera():
    """Adjust the camera position based on lil guy"""
    global camera_x
    if player_pos[0] > WIDTH * 0.7:
        camera_x += player_speed
        player_pos[0] -= player_speed
    elif player_pos[0] < WIDTH * 0.3 and camera_x:
        camera_x -= player_speed
        player_pos[0] += player_speed

def detect_collision(player, platform):
    """Detect if two rectangles are colliding."""
    return (
        player[0] < platform[0] + platform[2]
        and player[0] + player[2] > platform[0]
        and player[1] < platform[1] + platform[3]
        and player[1] + player[3] > platform[1]
    )

def check_on_ground_or_platform(): 
    """Check if player is on platform or ground"""
    global on_ground
    player_rect = [player_pos[0] + camera_x, player_pos[1] + player_size, player_size, 1]

    lowest_y = None
    for platform in platforms:
        platform_rect = [platform["x"], platform["y"], platform["width"], platform["height"]]
        if detect_collision(player_rect, platform_rect):
            if lowest_y is None or platform["y"] < lowest_y:
                lowest_y = platform["y"]
                if not platform["visited"]:
                    platform["visited"] = True
                    update_score(5)
    
    if lowest_y is None and player_pos[1] + player_size >= HEIGHT - ground_height:
        return HEIGHT - ground_height
    
    return lowest_y

def update_score(points):
    """Updates player score based on game conditions"""
    global score
    score += points


generate_platforms(0)

while not game_over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        player_pos[0] -= player_speed
    if keys[pygame.K_RIGHT]:
        player_pos[0] += player_speed
    if keys[pygame.K_SPACE] and on_ground:
        player_velocity_y = jump_strength
        on_ground = False

    player_velocity_y += gravity
    player_pos[1] += player_velocity_y

    new_y = check_on_ground_or_platform()
    if new_y is not None and player_pos[1] + player_size >= new_y:
        player_pos[1] = new_y - player_size
        player_velocity_y = 0
        on_ground = True
    else:
        on_ground = False

    handle_camera()

    if camera_x + WIDTH > furthest_x - WIDTH:
        generate_platforms(camera_x)

    platforms = [p for p in platforms if p["x"] + p["width"] > camera_x - WIDTH]

    screen.fill(GREY)

    pygame.draw.rect(screen, LIGHT_GREY, (0 - camera_x, HEIGHT - ground_height, camera_x + WIDTH, ground_height))
    for platform in platforms:
        pygame.draw.rect(screen, WHITE, (platform["x"] - camera_x, platform["y"], platform["width"], platform["height"]))
    pygame.draw.rect(screen, PINK, (player_pos[0], player_pos[1], player_size, player_size))

    score_text = font.render(f'Score: {score}', True, WHITE)
    screen.blit(score_text, (10, 10))

    pygame.display.flip()

    clock.tick(30)

pygame.quit()


