import pygame


pygame.init()

window = pygame.display.set_mode((800,800))

x = 0
y = 400

g = 0.002

yv = 0
xv = 0

plat_x = 0
plat_y = 300
check_c = 20

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    if keys[pygame.K_w]:
        y += 1
    if keys[pygame.K_a]:
            x -= 1
    if keys[pygame.K_s]:
        y -= 1
    if keys[pygame.K_d]:
        if x < 400:
            x += 1
        else:
            plat_x += 1
    
    yv -= g
    y += yv

    if y < plat_y:
        y = plat_y
        yv = 0


    window.fill((0, 0, 0))
    pygame.draw.rect(window, (255, 0, 0), (x - 25, 800 - y - 50, 50, 50))
    for i in range(check_c + 2):
        color = (255, 255, 0) if i%2==0 else (255, 255, 255)
        plat_x += .001
        pygame.draw.rect(window, color, (i * 800 / check_c - plat_x % (1600 / check_c), 800 - plat_y, 800 / check_c, plat_y))
    pygame.display.flip()

pygame.quit()
