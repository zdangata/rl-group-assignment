import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import numpy as np
from gym.wrappers.record_video import RecordVideo
from matplotlib import pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory():
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )
        
        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1)
                )
        
    def forward(self):
        raise NotImplementedError
    
    # for interacting with environment
    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        
        return action.item()
    
    # for ppo update
    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.value_layer(state)
        # action_logprobs indirectly represents the policy $\pi_{\theta}(s,a)$
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO():
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.ent_coef = 0.01

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def update(self, memory, timestep, total_timestep):   
        # Monte Carlo estimate of state rewards (can be replaced by General Advantage Estimators)
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # learning rate annealing
            frac = (timestep - 1.0) / total_timestep
            new_lr = self.lr * (1.0 - frac)
            new_lr = max(new_lr, 0.0)
            self.optimizer.param_groups[0]["lr"] = new_lr

            # entropy decay
            # self.ent_coef = max(0.001, self.ent_coef * (1.0 - frac))

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Finding Surrogate Loss (no gradient in advantages)
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            # MseLoss is for the update of critic, dist_entropy denotes an entropy bonus
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - self.ent_coef*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)  # Gradient clipping
            self.optimizer.step()

    
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
def main():
    ############## Hyperparameters ##############
    env_name = "LunarLander-v2"
    # creating environment
    env = gym.make(env_name, render_mode="rgb_array")
    # env = RecordVideo(env=env, video_folder="./videos", name_prefix="test-video")
    state_dim = env.observation_space.shape[0]
    action_dim = 4
    render = False
    solved_reward = 230         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 2000        # max training episodes
    max_timesteps = 500         # max timesteps in one episode
    n_latent_var = 256           # number of variables in hidden layer
    update_timestep = 2000      # update policy every n timesteps; batch timesteps
    lr = 0.003
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 8                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = 42
    #############################################
    
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    print('learning rate:',lr, 'Adam betas:', betas)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0

    rew_for_plots = 0
    rew_for_plots_list = []
    total_timestep = max_episodes * max_timesteps
    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()[0]
        
        for t in range(max_timesteps):
            timestep += 1
            
            # Running policy_old:
            # also append state, action, action_logprobs to the memory
            with torch.no_grad():
                action = ppo.policy_old.act(state, memory)
            state, reward, done, truncated, _ = env.step(action)
            
            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory, timestep, total_timestep)
                memory.clear_memory()
                timestep = 0
                break
            
            running_reward += reward
            rew_for_plots += reward

            # if render:
            #     env.render()
            if done or truncated:
                break
                
        avg_length += t
        rew_for_plots_list.append(rew_for_plots)
        rew_for_plots = 0

        
        # stop training if avg_reward > solved_reward
        # if running_reward > (log_interval*solved_reward):
        #     print("########## Solved! ##########")
        #     break

        if timestep >= total_timestep:
            break
        
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            # print(rew_for_plots)
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
    
        


    # def running_mean(x):
    #     N=50
    #     kernel = np.ones(N)
    #     conv_len = x.shape[0]-N
    #     y = np.zeros(conv_len)
    #     for i in range(conv_len):
    #         y[i] = kernel @ x[i:i+N]
    #         y[i] /= N
    #     return y
    

    def running_mean(x, N=50):
        kernel = np.ones(N) / N
        return np.convolve(x, kernel, mode='same')

    rew_for_plots_list = np.array(rew_for_plots_list)
    avg_score = running_mean(rew_for_plots_list)
    # episode_rewards_history_rand = np.array(episode_rewards_history_rand)  # random agent
    # avg_score_rand = running_mean(episode_rewards_history_rand)

    plt.figure(figsize=(15,7))
    plt.ylabel("Rewards",fontsize=12)
    plt.xlabel("Episodes",fontsize=12)
    plt.plot(rew_for_plots_list, color='gray' , linewidth=1)
    plt.plot(avg_score, color='blue', linewidth=3,label = 'Running Average Score of PPO Policy')
    # plt.plot(avg_score_rand, color='orange', linewidth=3,label = 'Running Average Score of Random Policy')
    plt.axhline(y=200, color='r', linestyle='-',label = 'Solved')
    plt.scatter(np.arange(rew_for_plots_list.shape[0]),rew_for_plots_list, 
                color='green' , linewidth=0.3, label='Episode Rewards')
    plt.legend()

    plt.title('Rewards by Episode For PPO in Lunar Lander')
    plt.show()


if __name__ == '__main__':
    main()