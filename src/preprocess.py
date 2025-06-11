#!/usr/bin/env python
import gymnasium as gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from train import ActorCritic, DiffusionModel, DBCTrainer

class DiffusionModelNoBayesian(DiffusionModel):
    def compute_loss(self, state_action_pairs):
        return torch.tensor(0.0)

def evaluate_model(actor_critic, trainer, env, episodes=5, domain_shift=False):
    rewards = []
    for ep in range(episodes):
        s = env.reset()
        if isinstance(s, tuple):
            s = s[0]
        done = False
        ep_reward = 0
        while not done:
            a = actor_critic.act(s)
            if domain_shift:
                s_shifted = s + np.random.normal(0, 0.2, size=np.array(s).shape)
            else:
                s_shifted = s
            s, r, done, truncated, _ = env.step(a)
            if isinstance(s, tuple):
                s = s[0]
            done = done or truncated
            ep_reward += r
        rewards.append(ep_reward)
    return np.mean(rewards)

def compute_average_uncertainty(trainer, sample_states):
    states_tensor = torch.FloatTensor(sample_states)
    predicted_actions = trainer.actor_critic(states_tensor)
    state_action_pairs = torch.cat([states_tensor, predicted_actions], dim=-1)
    uncertainty = trainer.diffusion_model.estimate_uncertainty(state_action_pairs)
    return uncertainty.mean().item()

def experiment_ablation_study():
    print("Starting Experiment 3: Ablation Study on Bayesian Regularization and Steering Module")
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor_critic = ActorCritic(state_dim, action_dim)

    diffusion_model_full = DiffusionModel(input_dim=state_dim + action_dim)
    trainer_full = DBCTrainer(actor_critic, diffusion_model_full, bc_weight=1.0, diffusion_weight=1.0)

    diffusion_model_no_steering = DiffusionModel(input_dim=state_dim + action_dim)
    trainer_no_steering = DBCTrainer(actor_critic, diffusion_model_no_steering, bc_weight=1.0, diffusion_weight=0.0)

    diffusion_model_no_bayes = DiffusionModelNoBayesian(input_dim=state_dim + action_dim)
    trainer_no_bayes = DBCTrainer(actor_critic, diffusion_model_no_bayes, bc_weight=1.0, diffusion_weight=1.0)

    eval_results = {}

    for variant_name, trainer_variant in zip(["Full", "No Steering", "No Bayesian"],
                                              [trainer_full, trainer_no_steering, trainer_no_bayes]):
        reward_normal = evaluate_model(actor_critic, trainer_variant, env, episodes=5, domain_shift=False)
        reward_shifted = evaluate_model(actor_critic, trainer_variant, env, episodes=5, domain_shift=True)
        eval_results[variant_name] = {"Normal": reward_normal, "Domain Shift": reward_shifted}
        print(f"{variant_name} Variant: Normal Domain Reward = {reward_normal:.2f}, Domain Shift Reward = {reward_shifted:.2f}")

    variants = list(eval_results.keys())
    normal_rewards = [eval_results[k]["Normal"] for k in variants]
    shifted_rewards = [eval_results[k]["Domain Shift"] for k in variants]

    x = np.arange(len(variants))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, normal_rewards, width, label="Normal Domain")
    plt.bar(x + width/2, shifted_rewards, width, label="Domain Shift")
    plt.xticks(x, variants)
    plt.ylabel("Average Reward")
    plt.title("Ablation Study: Domain Performance")
    plt.legend()
    plt.savefig("evaluation_ablation.pdf", bbox_inches="tight")
    plt.close()

    sample_states = np.random.rand(10, state_dim)
    for variant_name, trainer_variant in zip(["Full", "No Steering", "No Bayesian"],
                                              [trainer_full, trainer_no_steering, trainer_no_bayes]):
        avg_uncert = compute_average_uncertainty(trainer_variant, sample_states)
        print(f"{variant_name} variant average uncertainty: {avg_uncert:.4f}")

    print("Experiment 3 finished. Plot saved as evaluation_ablation.pdf.\n")
    
    env.close()
