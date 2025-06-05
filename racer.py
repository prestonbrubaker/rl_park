import pygame
import random
import torch

pygame.init()
window = pygame.display.set_mode((800,800))

font = pygame.font.SysFont("roboto", 18)

rounds = 10000

running = True

num_steps = 50000

games = []
labels = []

for round in range(rounds):
    x = random.randint(0,800)
    y = random.randint(0,800)

    # "sun"
    u = random.randint(0,800)
    v = random.randint(0,800)

    score = 0
    itC = 0

    game = []
    label = []

    while itC < num_steps:
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         itC = num_steps

        window.fill((0, 0, 0))

        r = random.randint(0,3)
        r_nn = random.randint(0,3)

        tmp_score = score

        # This is da sun
        if r == 0 and u<800:
            u+=1
        if r == 1 and v<800:
            v+=1
        if r == 2 and u>0:
            u-=1
        if r == 3 and v>0:
            v-=1

        # This is da lil guy
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or r_nn == 0:
            if y>0:
                y -= 1
        if keys[pygame.K_a] or r_nn == 1:
            if x>0:
                x -= 1
        if keys[pygame.K_s] or r_nn == 2:
            if y<800:
                y += 1
        if keys[pygame.K_d] or r_nn == 3:
            if x<800:
                x += 1
        
        pygame.draw.rect(window, (255, 255, 255), (u-40,v-40,80,80))

        pygame.draw.rect(window, (255, 0, 0), (x-10,y-10,20,20))

        if x>u-10 and x<u+10 and y>v-10 and y<v+10:
            score += 1

        dist = ((x - u) ** 2 + (y - v) ** 2) ** 0.5

        score -= dist / 100000

        text = font.render(str(score), True, (0, 255, 0))
        text_rect = text.get_rect(center=(100, 20))

        window.blit(text, text_rect)
        itC += 1

        game_stat = [0] * 4
        game_stat[r_nn] = 1
        game_stat.append(u - x)
        game_stat.append(v - y)
        game_stat.append(x)
        game_stat.append(y)

        label_stat = [(score - tmp_score)]


        print(game_stat)

        print(f"Score: {label_stat}\n")
        
        game.append(game_stat)
        label.append(label_stat)

        pygame.display.flip()
    
    games.append(game)
    labels.append(label)

    games_t = torch.tensor(games).float()
    labels_t = torch.tensor(labels).float()        
    torch.save(games_t, "games.pth")
    torch.save(labels_t, "labels.pth")

pygame.quit()
