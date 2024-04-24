import torch
import torch.nn as nn
import numpy as np

from torch.distributions import MultivariateNormal
from network import FeedForwardNN
from torch.optim import Adam
from matplotlib import pyplot as plt

class PPO:
    def __init__(self, env):
        # extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        # self.act_dim = env.action_space.shape[0]  ## for environments where the action space is box 
        self.act_dim = env.action_space.n ## for environments where the action space is discrete 
        
        self._init_hyperparameters()

        # Algorithm step 1
        # initialise actor and critic networks
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

        # Create variable for matrix. 0.5 is chosen of stdev arbitrarily
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var) 

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
            
    def _init_hyperparameters(self):
        # Default values for hyperparameters, will need to change later
        self.timesteps_per_batch = 4800                 # timesteps per batch
        self.max_timesteps_per_episode = 1600           # timesteps per episode
        self.gamma = 0.95                               # discount factor
        self.n_updates_per_iteration = 5                # number of epochs per iteration
        self.clip = 0.2                                 # as recommended by the paper
        self.lr = 0.005                                 # learning rate of optimiser

    
    def rollout(self): 
        """
        In the rollout function, it collects batch data and stores it.
        In each batch timestep, there will be an inner loop which executes depending on the amount 
        of timesteps definede per episode. 
        In each global timestep, this function will be executed. 
        """
        # Batch data
        batch_obs = []          # batch observations
        batch_acts = []         # batch actions
        batch_log_probs = []    # log probs of each action
        batch_rews = []          # batch rewards
        batch_rtgs = []         # batch rewards to go 
        batch_lens = []         # episode lengths in batch
               
        # number of timesteps run so far this batch
        t = 0
        
        while t < self.timesteps_per_batch:
            # rewards this episode
            ep_rews = []

            obs, _ = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):   # initiate the rollout for an episode
                t += 1  # increment timesteps ran this batch so far

                # collect observation
                batch_obs.append(obs)
                

                action, log_prob = self.get_action_for_discrete(obs)


                obs, rew, done, _, _ = self.env.step(action)

                # collect reward, action, and log prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break
            
            # collect episodic length and rewards
            batch_lens.append(ep_t + 1)  # plus 1 because timestep starts at 0
            batch_rews.append(ep_rews)
        
        # reshape the data as tensors before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        # Algorithm step 4
        batch_rtgs = self.compute_rtgs(batch_rews)
        
        # return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens , batch_rews

    
    def get_action_for_discrete(self, obs):
        probabilities = self.actor(obs)
        dist = torch.distributions.Categorical(probabilities)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().item(), log_prob.detach()


    def get_action(self, obs):
        # query the actor network for mean action
        mean = self.actor(obs)
        print(f"MEAN -> {mean}")
        # Create our Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)
        print(f"DIST -> {dist}")
        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # return the sampled action and its log prob
        # we use detach() as they are tensors with computation graphs.
        # so we detach the graphs and convert the action to a numpy array but keep log prob as tensor
        # Our computation graph will start later
        return action.detach().numpy(), log_prob.detach()
    

    def compute_rtgs(self, batch_rews):
        # the rtg per episode per batch to return
        # the shape will be (num timesteps per episode)
        batch_rtgs = []

        # iterate through each episode BACKWARDS to maintain same order in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0  # the discounted rew so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
            
        # convert the rtgs into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs
    

    def learn(self, total_timesteps):
        t_so_far = 0  # timesteps simulated so far, this allows you to specify the amount of timesteps to train in total
        summed_rewards = 0
        summed_rewards_list = []
        while t_so_far < total_timesteps:           # Algorithm step 2 
            # Algorithm step 3
            # batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, rewards = self.rollout()
            print(t_so_far)
            # summed_rewards += rewards
            summed_rewards_list.append(rewards)

            # calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Calculate V {phi, k} for advantage 
            # V, _ = self.evaluate(batch_obs, batch_acts)
            V, _ = self.new_evaluate(batch_obs, batch_acts)

            # Algorithm step 5
            # Calculate advantage
            A_k = batch_rtgs - V.detach()

            # Normalise advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)  # normalising very impt in practice altho not included explicitly in pseudocode, w/o it training highly unstable

            for _ in range(self.n_updates_per_iteration):
                # calculate pi_theta(a_t | s_t)
                # V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                V, curr_log_probs = self.new_evaluate(batch_obs, batch_acts)

                # calculate ratios
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                # calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # calculate actor and critic loss
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # calculate gradients and perform backpropagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)  # we need to add retain_graph=True for the first network we backpropagate on or we'll get an error
                self.actor_optim.step()

                # calculate gradients and perform backpropagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

        return summed_rewards_list
    

    def evaluate(self, batch_obs, batch_acts):
        # query critic network for a value V for each obs in batch_obs
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probs of batch actions using most recent actor network
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        # return predicted values V and log probs log_probs
        return V, log_probs
    

    def new_evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()

        probs = self.actor(batch_obs)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs
    
     
import gym
env = gym.make("CartPole-v1")


model = PPO(env)
summed_rewards_list = model.learn(100000)
# print(f"summed rewards list {summed_rewards_list}")
print(f"summed rewards list length - > {len(summed_rewards_list)}")

total_avg_rews = []

for batch_rews in summed_rewards_list:
    avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in batch_rews])
    total_avg_rews.append(avg_ep_rews)


# x_values = np.arange(len(total_avg_rews))
# print(x_values)
# scaled_x_values = x_values * 1600 * 3
# print(scaled_x_values)

# plt.figure(figsize=(15,7))
# plt.plot(scaled_x_values, total_avg_rews)
plt.plot(total_avg_rews)
plt.xlabel("Number of batches")
plt.ylabel("Average rewards per batch")
plt.show()