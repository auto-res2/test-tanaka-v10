#!/usr/bin/env python
import gymnasium as gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Linear(state_dim, 128)
        self.action_head = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc(x))
        action_logits = self.action_head(x)
        return action_logits

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits = self.forward(state_tensor)
        action = torch.argmax(logits, dim=-1).item()
        return action

class DiffusionModel(nn.Module):
    def __init__(self, input_dim):
        super(DiffusionModel, self).__init__()
        self.fc = nn.Linear(input_dim, 64)
        self.out = nn.Linear(64, 1)

    def compute_loss(self, state_action_pairs):
        x = F.relu(self.fc(state_action_pairs))
        energy = self.out(x)
        loss = F.mse_loss(energy, torch.zeros_like(energy))
        return loss

    def estimate_uncertainty(self, state_action_pairs):
        with torch.no_grad():
            energy = self.out(F.relu(self.fc(state_action_pairs)))
            uncertainty = torch.abs(energy)
        return uncertainty

class DBCTrainer:
    def __init__(self, actor_critic, diffusion_model, bc_weight=1.0, diffusion_weight=1.0, optimizer=None):
        self.actor_critic = actor_critic
        self.diffusion_model = diffusion_model
        self.bc_weight = bc_weight
        self.diffusion_weight = diffusion_weight
        if optimizer is None:
            self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=1e-3)
        else:
            self.optimizer = optimizer

    def update(self, states, expert_actions):
        states_tensor = torch.FloatTensor(states)
        predicted_actions = self.actor_critic(states_tensor)

        expert_actions_tensor = torch.FloatTensor(expert_actions)
        bc_loss = F.mse_loss(predicted_actions, expert_actions_tensor)

        state_action_pairs = torch.cat([states_tensor, predicted_actions], dim=-1)
        diffusion_loss = self.diffusion_model.compute_loss(state_action_pairs)

        loss = self.bc_weight * bc_loss + self.diffusion_weight * diffusion_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        uncertainty = self.diffusion_model.estimate_uncertainty(state_action_pairs)
        return loss.item(), uncertainty.mean().item()

def experiment_domain_shift():
    print("Starting Experiment 1: Domain Shift Detection and Adaptation Test")
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor_critic = ActorCritic(state_dim, action_dim)
    diffusion_model = DiffusionModel(input_dim=state_dim + action_dim)
    trainer = DBCTrainer(actor_critic, diffusion_model)

    num_episodes = 50
    domain_shift_episode = 25

    performance_metrics = []
    uncertainty_over_time = []

    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        done = False
        episode_reward = 0
        while not done:
            action = actor_critic.act(state)

            if episode >= domain_shift_episode:
                noisy_state = state + np.random.normal(0, 0.2, size=np.array(state).shape)
            else:
                noisy_state = state

            next_state, reward, done, truncated, _ = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            done = done or truncated
            episode_reward += reward

            expert_action = np.eye(action_dim)[action]
            loss, uncertainty = trainer.update([noisy_state], [expert_action])
            uncertainty_over_time.append(uncertainty)
            state = next_state

        performance_metrics.append(episode_reward)
        print(f"Episode {episode+1}, Reward: {episode_reward}")

    plt.figure()
    plt.plot(performance_metrics, label="Episode Reward")
    plt.axvline(domain_shift_episode, color='r', linestyle='--', label="Domain Shift Start")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Performance before and after Domain Shift")
    plt.legend()
    plt.savefig("performance_domain_shift.pdf", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(uncertainty_over_time, label="Bayesian Uncertainty")
    plt.xlabel("Update Steps")
    plt.ylabel("Uncertainty")
    plt.title("Uncertainty Estimation Over Time")
    plt.legend()
    plt.savefig("uncertainty_domain_shift.pdf", bbox_inches="tight")
    plt.close()
    print("Experiment 1 finished. Plots saved as performance_domain_shift.pdf and uncertainty_domain_shift.pdf.\n")
    
    env.close()
