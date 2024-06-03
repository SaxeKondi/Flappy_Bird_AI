import pygame
import neat
import time
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil
from GeneticAlgorithm import *
from DeepQLearning import DQNAgent
pygame.font.init()

def get_movement_characteristics(max_jump_height, jump_peak_time):
    # jump_height = v*t + a*(0.5*t^2 - 0.5*t)
    # eq1: v*jump_peak_time + a*(0.5*jump_peak_time^2 - 0.5*jump_peak_time) = max_jump_height
    # eq2: v + a*(jump_peak_time - 0.5) = 0
    a = np.array([[jump_peak_time, 0.5 * jump_peak_time**2 - 0.5 * jump_peak_time], [1, jump_peak_time - 0.5]])
    b = np.array([max_jump_height, 0])
    [v, acc] = np.linalg.solve(a,b)
    return -v, -acc #negated because higher pixel positions have lower values

FPS = 60

WIN_WIDTH = 600
WIN_HEIGHT = WIN_WIDTH * 4 / 3
SCALE = WIN_WIDTH / 444

GROUND_HEIGHT = round(WIN_HEIGHT - SCALE * 73)

PIPE_WIDTH = floor(SCALE * 80)
PIPE_MIN_SIZE = round(SCALE * 74)
PIPE_OPENING = ceil(SCALE * 144)
PIPE_SPACING = PIPE_WIDTH + ceil(SCALE * 182)

BIRD_START_X = floor(SCALE * 169)
BIRD_START_Y = round(SCALE * 288)
BIRD_WIDTH = floor(SCALE * 53)
BIRD_HEIGHT = floor(SCALE * 40)

BIRD_TERM_VEL = round(SCALE / FPS * 720) #(px/s -> px/frame)
JUMP_HEIGHT = floor(SCALE * 70)
JUMP_TIME = round(FPS / 4)
JUMP_VEL_INIT, GRAVITY = get_movement_characteristics(JUMP_HEIGHT, JUMP_TIME)

ENV_VEL = round(SCALE / FPS * 210) #(px/s -> px/frame)

BIRD_TILT_DOWN = (SCALE * 20)

GEN = 0

win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))

BIRD_IMGS = [pygame.transform.smoothscale(pygame.image.load(os.path.join("imgs", "bird1.png")).convert_alpha(),(BIRD_WIDTH,BIRD_HEIGHT)),
            pygame.transform.smoothscale(pygame.image.load(os.path.join("imgs", "bird2.png")).convert_alpha(),(BIRD_WIDTH,BIRD_HEIGHT)),
            pygame.transform.smoothscale(pygame.image.load(os.path.join("imgs", "bird3.png")).convert_alpha(),(BIRD_WIDTH,BIRD_HEIGHT))]
PIPE_BOTTOM_IMG = pygame.transform.smoothscale_by(pygame.image.load(os.path.join("imgs", "pipe.png")).convert_alpha(),
                                            PIPE_WIDTH / pygame.image.load(os.path.join("imgs", "pipe.png")).get_width())
PIPE_TOP_IMG = pygame.transform.flip(PIPE_BOTTOM_IMG, False, True)
# BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird1.png"))),
#             pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird2.png"))), 
#             pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird3.png")))]
# PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))
GROUND_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))

PIPE_TOP_MASK = pygame.mask.from_surface(PIPE_TOP_IMG)
PIPE_BOTTOM_MASK = pygame.mask.from_surface(PIPE_BOTTOM_IMG)
BIRD_MASKS = [pygame.mask.from_surface(BIRD_IMGS[0]), pygame.mask.from_surface(BIRD_IMGS[1]), pygame.mask.from_surface(BIRD_IMGS[2])]

STAT_FONT = pygame.font.SysFont("comicsans", 50)


class Bird:
    IMGS = BIRD_IMGS
    MASKS = BIRD_MASKS
    MAX_ROTATION = 25
    ROT_VEL = FPS * 2 / 3
    ANIMATION_TIME = int(round(FPS / 6))

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.acc = GRAVITY
        self.vel = 0
        self.jump_pos = self.y
        self.img_count = 0
        self.img = self.IMGS[0]
        self.animation_index = 0

    def jump(self):
        self.vel = JUMP_VEL_INIT
        self.tick_count = 0
        self.jump_pos = round(self.y)

    def move(self):
        if self.vel > BIRD_TERM_VEL:
            self.vel = BIRD_TERM_VEL

        self.y = self.y + self.vel

        self.vel = self.vel + self.acc

        if self.vel < 0 or round(self.y) < self.jump_pos + BIRD_TILT_DOWN:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION

        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL
    
    def draw(self, win):
        self.animation_index = (self.img_count // self.ANIMATION_TIME) % len(self.IMGS)
        self.img = self.IMGS[self.animation_index]
        self.img_count = (self.img_count + 1) % (self.ANIMATION_TIME * len(self.IMGS))

        if self.tilt <= -80:
            self.animation_index = 1
            self.img = self.IMGS[self.animation_index]
            self.img_count = self.ANIMATION_TIME*2

        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center = self.img.get_rect(topleft = (self.x, round(self.y))).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return self.MASKS[self.animation_index]
    
class Pipe:
    OPENING = PIPE_OPENING
    VEL = ENV_VEL

    def __init__(self, x, random_generator: np.random.Generator):
        self.x = x
        self.height = 0

        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = PIPE_TOP_IMG
        self.PIPE_BOTTOM = PIPE_BOTTOM_IMG

        self.top_mask = PIPE_TOP_MASK
        self.bottom_mask = PIPE_BOTTOM_MASK

        self.passed = False
        self.set_height(random_generator)

    def set_height(self, random_generator: np.random.Generator):
        self.height = random_generator.integers(PIPE_MIN_SIZE , GROUND_HEIGHT - PIPE_MIN_SIZE - PIPE_OPENING, endpoint= True) 
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.OPENING

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird: Bird):
        bird_mask = bird.get_mask()

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        t_point = bird_mask.overlap(self.top_mask, top_offset)
        b_point = bird_mask.overlap(self.bottom_mask, bottom_offset)

        if(t_point or b_point):
            return True
        
        return False
    
class Ground:
    VEL = ENV_VEL
    WIDTH = GROUND_IMG.get_width()
    IMG = GROUND_IMG

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))



def draw_window(birds, pipes, ground, score, gen, alive):
    global win
    win.blit(BG_IMG, (0,0))

    for pipe in pipes:
        pipe.draw(win)

    text = STAT_FONT.render(f"Score: {score}", 1, (255,255,255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

    text = STAT_FONT.render(f"Alive: {alive}", 1, (255,255,255))
    win.blit(text, (10, 10))

    if gen > 0:
        text = STAT_FONT.render(f"Gen: {gen}", 1, (255,255,255))
        win.blit(text, (10, 60))

    ground.draw(win)
    for bird in birds:
        bird.draw(win)
    pygame.display.update()

##############################################################################

class Game():
    global win
    def __init__(self, seed = np.random.randint(1000000)):
        self.generator = np.random.default_rng(seed= seed)
        self.action_space = np.array([0,1])
        self.bird = Bird(230,350)
        self.ground = Ground(GROUND_HEIGHT)
        self.pipes = [Pipe(WIN_WIDTH + PIPE_SPACING, self.generator)]
        self.score = 0
        self.done = False

    def get_state(self):
        pipe_ind = 0
        if len(self.pipes) > 1 and self.bird.x > self.pipes[0].x + self.pipes[0].PIPE_TOP.get_width():
            pipe_ind = 1
        # vert_dist_top = abs(self.bird.y - self.pipes[pipe_ind].height)
        # vert_dist_bottom = abs(self.bird.y - self.pipes[pipe_ind].bottom)
        vert_dist_middle = self.pipes[pipe_ind].height + PIPE_OPENING/2 - self.bird.y
        hori_dist_pipe = self.pipes[pipe_ind].x + self.pipes[pipe_ind].PIPE_TOP.get_width() - self.bird.x

        # return [self.bird.y, vert_dist_top, vert_dist_bottom, hori_dist_pipe, self.bird.vel] 
        return [self.bird.y, vert_dist_middle, hori_dist_pipe, self.bird.vel] 
        # return [self.bird.y / GROUND_HEIGHT, vert_dist_middle / (GROUND_HEIGHT - PIPE_MIN_SIZE - PIPE_OPENING/2), hori_dist_pipe / PIPE_SPACING, self.bird.vel / BIRD_TERM_VEL] #normalised
        

    def reset(self):
        self.bird = Bird(230,350)
        self.ground = Ground(GROUND_HEIGHT)
        self.pipes = [Pipe(WIN_WIDTH + PIPE_SPACING, self.generator)]
        self.score = 0
        self.done = False
        return self.get_state()

    def step(self, action):
        reward = 0.05
        # reward = 0
        if action > 0.5:
            self.bird.jump()

        self.bird.move()

        for pipe in self.pipes:
            pipe.move()

            if pipe.collide(self.bird):
                reward = -1
                self.done = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                j = self.pipes.index(pipe)
                self.pipes.pop(j)

        if len(self.pipes) < 2 or self.pipes[-1].x <= WIN_WIDTH or self.pipes[-2].x <= self.bird.x:
            self.pipes.append(Pipe(self.pipes[-1].x + PIPE_SPACING, self.generator))
        
        if not self.pipes[0].passed and self.pipes[0].x + self.pipes[0].PIPE_TOP.get_width() < self.bird.x:
            self.pipes[0].passed = True 
            self.score += 1
            reward = 5

        if round(self.bird.y) + self.bird.img.get_height() >= GROUND_HEIGHT or round(self.bird.y) + self.bird.img.get_height() < 0:
            reward = -1
            self.done = True
        
        if self.score >= 200:
            self.done = True

        self.ground.move()

        return self.get_state(), reward, self.done, self.score


class GA_Env():
    def __init__(self):
        self.scores = []
        self.maxScore = []
        self.avgScore = []

        self.fitnesses = []
        self.maxFitness = []
        self.avgFitness = []

        self.gen = []
        self.episodes = []

    def fitnessFunction(self, genomes: np.ndarray, neuralNetwork: neuralNetwork):
        seed = np.random.randint(1000000)
        ge = []
        games = []
        scores = []
        fitnesses = []
        global GEN
        GEN += 1
        for genome in genomes:
            games.append(Game(seed= seed))
            genome.fitness = 0
            ge.append(genome)
        
        while(len(ge) > 0):
            pygame.event.pump()
            draw_window([game.bird for game in games], games[0].pipes, games[0].ground, games[0].score, GEN, len(games)) #draw all birds
            # draw_window([games[0].bird], games[0].pipes, games[0].ground, games[0].score, GEN, len(games)) #draw one bird
            for game in games:
                i = games.index(game)
                _, reward, done, score = game.step(neuralNetwork.feedForward(ge[i], game.get_state()[:neuralNetwork.architecture[0]]))
                ge[i].fitness += reward
                if done:
                    scores.append(score)
                    fitnesses.append(ge[i].fitness)
                    ge.pop(i)
                    games.pop(i)
                    

        self.scores.append(scores)
        self.maxScore.append(max(scores))
        self.avgScore.append(sum(scores) / len(scores))
        self.fitnesses.append(fitnesses)
        self.maxFitness.append(max(fitnesses))
        self.avgFitness.append(sum(fitnesses) / len(fitnesses))
        self.gen.append(len(self.scores))
        self.episodes.append(len(self.scores)*len(self.scores[0]))

def run_genetic():
    env = GA_Env()
    nn = neuralNetwork([4,5,1]) #input layer should be min 2 and max 4. output layer should be 1
    # nn = neuralNetwork([2,2,1]) #input layer should be min 2 and max 4. output layer should be 1
    GeneticAlgorithm(env.fitnessFunction, 1700, nn, populationSize= 50, maxGenerations=5000, elitism=3)

    plt.plot(env.gen, env.maxScore, label= "Max score")
    plt.plot(env.gen, env.avgScore, label= "Average score")
    plt.legend(loc="upper left")
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.show()

    plt.plot(env.episodes, env.maxScore, label= "Max score")
    plt.plot(env.episodes, env.avgScore, label= "Average score")
    plt.legend(loc="upper left")
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.show()

    plt.plot(env.gen, env.maxFitness, label= "Max fitness")
    plt.plot(env.gen, env.avgFitness, label= "Average fitness")
    plt.legend(loc="upper left")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.show()

    plt.plot(env.episodes, env.maxFitness, label= "Max fitness")
    plt.plot(env.episodes, env.avgFitness, label= "Average fitness")
    plt.legend(loc="upper left")
    plt.xlabel("Episodes")
    plt.ylabel("Fitness")
    plt.show()


class DQN_Env():
    def __init__(self, state_num):
        self.game = Game()
        self.state_num = state_num
        self.episodes = 0

    def reset(self):
        state = self.game.reset()[:self.state_num]
        return state

    def step(self, action):
        # pygame.event.pump()
        # draw_window([self.game.bird], self.game.pipes, self.game.ground, self.game.score, GEN, 0)
        state, reward, done, score = self.game.step(action)
        if(done):
            self.episodes += 1
            print(self.episodes)
        return state[:self.state_num], reward, done
    

def run_deep_q():
    state_num = 4
    env = DQN_Env(state_num)
    agent = DQNAgent(env, state_size=state_num, gamma = 0.99, lr=5e-4, memory_size=10000)
    agent.train(episodes=10000, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, target_update=10)

    
def run_manual():
    game = Game()
    clock = pygame.time.Clock()
    clock.tick(FPS)
    run = True
    done = False
    action = 0
    while run:
        draw_window([game.bird], game.pipes, game.ground, game.score, GEN, 0) #draw one bird
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action = 1
        if not done:
            done = game.step(action)[2]
            action = 0
        elif action == 1:
            game.reset()
            done = False

if __name__ == "__main__":
    run_genetic()
    # run_deep_q()
    # run_manual()




