import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

import matplotlib.pyplot as plt

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQNAgent:
    def __init__(self, env, state_size, action_size = 2, batch_size=64, gamma=0.99, lr=0.001, memory_size=10000):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr

        self.memory = ReplayMemory(memory_size)
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        self.scoreLog = []

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, epsilon=0.0):
        if np.random.rand() <= epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        current_q_values = self.model(states).gather(1, actions).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = self.loss_fn(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, episodes, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, target_update=10):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()

        epsilon = epsilon_start
        for e in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.act(state, epsilon)
                next_state, reward, done = self.env.step(action)
                self.memory.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                self.replay()
            if e % target_update == 0:
                self.update_target_model()
                self.scoreLog.append(self.evaluate())
                print(self.scoreLog)

                ax.clear()
                ax.plot(self.scoreLog)
                plt.draw()
                plt.pause(0.01)

            epsilon = max(epsilon_end, epsilon_decay * epsilon)

        plt.ioff()
        plt.show()

    def evaluate(self, episodes=10):
        total_rewards = []
        for e in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.act(state, epsilon=0.0)  # No exploration during evaluation
                state, reward, done = self.env.step(action)
                total_reward += reward
            total_rewards.append(total_reward)
        average_reward = np.mean(total_rewards)
        return average_reward