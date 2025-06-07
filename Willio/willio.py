import pygame
import random
import time

pygame.init()

window = pygame.display.set_mode((800,800))

font = pygame.font.SysFont("roboto", 18)

x = 0
y = 400

g = 0.06
sens = 5

yv = 0
xv = 0

plat_x = 0
plat_y = 300
check_c = 10

score = 0


a = []
for i in range(1000):
    r = random.uniform(0, 1)
    r = 0 if r <0.9 else 1
    a.append(r)


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    x_old = x + plat_x

    if keys[pygame.K_w] and y < plat_y + 10:
        y += sens * .1
        yv += 0.8
    if keys[pygame.K_a]:
        if x > 100:
            x -= sens
        else:
            plat_x -= sens
    if keys[pygame.K_s]:
        y -= sens
    if keys[pygame.K_d]:
        if x < 400:
            x += sens
        else:
            plat_x += sens
    
    yv -= g
    y += yv

    if y < plat_y and not a[int((plat_x + x) / 800 * check_c)]:
        y = plat_y
        yv = 0
    

    ds = x + plat_x - x_old

    if y < plat_y:
        ds -= plat_y - y

    score += ds

    window.fill((0, 0, 0))
    
    for i in range(check_c + 2):
        color = (255, 255, 0) if (i + int(plat_x * check_c / 800))%2==0 else (255, 255, 255)
        color = (0, 0, 0) if a[i + int(plat_x / 800 * check_c)] else color
        plat_x += .001
        pygame.draw.rect(window, color, (i * 800 / check_c - plat_x % (800 / check_c), 800 - plat_y, 800 / check_c, plat_y))
    pygame.draw.rect(window, (255, 0, 0), (x - 25, 800 - y - 50, 50, 50))
    text = font.render(f"Score: {score:.2f}", True, (0, 255, 0))
    text_rect = text.get_rect(center=(100, 20))
    window.blit(text, text_rect)
    pygame.display.flip()
    time.sleep(0.01)

pygame.quit()
