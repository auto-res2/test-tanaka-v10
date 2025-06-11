#!/usr/bin/env python
import gymnasium as gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from train import ActorCritic, DiffusionModel, DBCTrainer

def online_bayesian_update(trainer, new_data_states, new_data_expert_actions):
    start_time = time.time()
    loss, _ = trainer.update(new_data_states, new_data_expert_actions)
    update_time = time.time() - start_time
    return loss, update_time

def full_retraining_update(actor_critic, diffusion_model, optimizer, new_data_states, new_data_expert_actions, epochs=2):
    states_tensor = torch.FloatTensor(new_data_states)
    expert_actions_tensor = torch.FloatTensor(new_data_expert_actions)
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        predicted_actions = actor_critic(states_tensor)
        bc_loss = F.mse_loss(predicted_actions, expert_actions_tensor)
        state_action_pairs = torch.cat([states_tensor, predicted_actions], dim=-1)
        diffusion_loss = diffusion_model.compute_loss(state_action_pairs)
        loss = bc_loss + diffusion_loss
        loss.backward()
        optimizer.step()
    update_time = time.time() - start_time
    return loss.item(), update_time

def experiment_online_update():
    print("Starting Experiment 2: Online Bayesian Update Efficiency")
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor_critic = ActorCritic(state_dim, action_dim)
    diffusion_model = DiffusionModel(input_dim=state_dim + action_dim)
    trainer = DBCTrainer(actor_critic, diffusion_model)

    bayesian_losses, bayesian_times = [], []
    full_retrain_losses, full_retrain_times = [], []

    for update_iter in range(10):
        new_data_states = np.random.rand(5, state_dim)
        new_data_expert_actions = np.eye(action_dim)[np.random.randint(0, action_dim, size=5)]

        loss_b, time_b = online_bayesian_update(trainer, new_data_states, new_data_expert_actions)
        loss_f, time_f = full_retraining_update(actor_critic, diffusion_model, trainer.optimizer, new_data_states, new_data_expert_actions)

        bayesian_losses.append(loss_b)
        bayesian_times.append(time_b)
        full_retrain_losses.append(loss_f)
        full_retrain_times.append(time_f)

        print(f"Iteration {update_iter+1}: Bayesian Update Loss={loss_b:.4f}, Time={time_b:.4f} sec; "
              f"Full Retraining Loss={loss_f:.4f}, Time={time_f:.4f} sec")

    plt.figure()
    plt.plot(bayesian_times, label="Bayesian Update Time")
    plt.plot(full_retrain_times, label="Full Retraining Time")
    plt.xlabel("Update Iteration")
    plt.ylabel("Time (sec)")
    plt.title("Comparison of Update Times")
    plt.legend()
    plt.savefig("inference_latency_online_update.pdf", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(bayesian_losses, label="Bayesian Update Loss")
    plt.plot(full_retrain_losses, label="Full Retraining Loss")
    plt.xlabel("Update Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Reduction Rate Comparison")
    plt.legend()
    plt.savefig("training_loss_online_update.pdf", bbox_inches="tight")
    plt.close()
    print("Experiment 2 finished. Plots saved as inference_latency_online_update.pdf and training_loss_online_update.pdf.\n")
    
    env.close()
