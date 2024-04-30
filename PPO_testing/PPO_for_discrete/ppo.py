import torch
import torch.nn as nn
import numpy as np

from torch.distributions import MultivariateNormal
from network import FeedForwardNN
from torch.optim import Adam
from matplotlib import pyplot as plt
from gym.wrappers.record_video import RecordVideo


class PPO:
    def __init__(self, env):
        # extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        # self.act_dim = env.action_space.shape[0]  ## for environments where the action space is box 
        self.act_dim = env.action_space.n ## for environments where the action space is discrete 
        
        self._init_hyperparameters()

        # Check for GPU availability and set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {'GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")


        # Algorithm step 1
        # initialise actor and critic networks
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim).to(self.device)
        self.critic = FeedForwardNN(self.obs_dim, 1).to(self.device)

        # Create variable for matrix. 0.5 is chosen of stdev arbitrarily
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5, device=self.device)
        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var) 

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
            
    def _init_hyperparameters(self):
        # Default values for hyperparameters, will need to change later
        self.timesteps_per_batch = 11200                 # timesteps per batch
        self.max_timesteps_per_episode = 1600           # timesteps per episode
        self.gamma = 0.999                               # discount factor
        self.n_updates_per_iteration = 6                # number of epochs per iteration
        self.clip = 0.2                                 # as recommended by the paper
        self.lr = 0.05                                 # learning rate of optimiser
        
        self.num_minibatches = 100                       # number of minibatch updates
        self.ent_coef = 0.1                               # Entropy coefficient for entropy regularisation
        self.max_grad_norm = 0.5                        # Gradient clipping threshold typically 0.5
        self.target_kl = 0.01                           # KL Divergence threshold
        self.lam = 0.98                                 # Lambda Parameter for GAE 

    
    def rollout(self, t_so_far): 
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
        batch_vals = []
        batch_dones = []
               
        # number of timesteps run so far this batch
        t = 0
        video_count = 0

        total_rew_per_ep = []
        avg_rew_per_ep = []
        cumulative_timesteps = []

        while t < self.timesteps_per_batch:
            ep_rews = []  # rewards collected per episode
            ep_vals = []  # state values collected per episode
            ep_dones = [] # done flag collected per episode


            obs, _ = self.env.reset()
            obs = torch.tensor(obs, dtype=torch.float).to(self.device)
            
            done = False
            episodic_return = []
            total_episodic_return = 0

            video_start = (self.timesteps_per_batch - t) <= 50

            if video_start:
                # Start the recorder
                self.env.start_video_recorder()

            for ep_t in range(self.max_timesteps_per_episode):   # initiate the rollout for an episode
                ep_dones.append(done)

                t += 1  # increment timesteps ran this batch so far
                t_so_far += 1

                # collect observation
                batch_obs.append(obs)

                action, log_prob = self.get_action_for_discrete(obs)
                val = self.critic(obs)

                obs, rew, done, truncated, _ = self.env.step(action)
                obs = torch.tensor(obs, dtype=torch.float).to(self.device)


                total_episodic_return += rew
                # episodic_return.append(rew)


                # collect reward, action, and log prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                ep_vals.append(val.flatten())

                if done or truncated:
                    break
            
            avg_rew_per_ep.append(np.mean(episodic_return))
            total_rew_per_ep.append(total_episodic_return)
            cumulative_timesteps.append(t_so_far)
            
            if video_start:
                # Don't forget to close the video recorder before the env!
                print("hihihihihihihi")
                self.env.close_video_recorder()
            
            # collect episodic length and rewards
            batch_lens.append(ep_t + 1)  # plus 1 because timestep starts at 0
            batch_rews.append(ep_rews)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)
        

        batch_obs = torch.stack(batch_obs).to(self.device)      

        # reshape the data as tensors before returning
        # batch_obs = torch.tensor(batch_obs, dtype=torch.float).to(self.device)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float).to(self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(self.device)

        # NO LONGER calculate RTGs        
        # # Algorithm step 4
        # batch_rtgs = self.compute_rtgs(batch_rews)
        
        # return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones, total_rew_per_ep, cumulative_timesteps

    
    def get_action_for_discrete(self, obs):
        probabilities = self.actor(obs)
        dist = torch.distributions.Categorical(probabilities)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().item(), log_prob.detach()


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
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).to(self.device)
        return batch_rtgs
    

    def learn(self, total_timesteps):
        t_so_far = 0  # timesteps simulated so far, this allows you to specify the amount of timesteps to train in total
        
        summed_rewards_list = []
        timestep_counter = []

        while t_so_far < total_timesteps:           # Algorithm step 2 
            # Algorithm step 3
            # batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones, avg_rew_per_ep, cumulative_timesteps = self.rollout(t_so_far)
            
            print(t_so_far)
            summed_rewards_list.extend(avg_rew_per_ep)
            timestep_counter.extend(cumulative_timesteps)
            

            A_k = self.calculate_gae(batch_rews, batch_vals, batch_dones)
            V = self.critic(batch_obs).squeeze()
            batch_rtgs = A_k + V.detach()

            # calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Calculate V {phi, k} for advantage 
            # V, _ = self.evaluate(batch_obs, batch_acts)
            # V, _, _ = self.new_evaluate(batch_obs, batch_acts)

            # # Algorithm step 5
            # # Calculate advantage
            # A_k = batch_rtgs - V.detach()

            # Normalise advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)  # normalising very impt in practice altho not included explicitly in pseudocode, w/o it training highly unstable

            # These are the variables needed for mini-batch updates
            step = batch_obs.size(0)
            inds = np.arange(step)
            minibatch_size = step // self.num_minibatches

            for _ in range(self.n_updates_per_iteration):
                # linear rate annealing
                frac = (t_so_far - 1.0) / total_timesteps
                new_lr = self.lr * (1.0 - frac)
                new_lr = max(new_lr, 0.0)
                self.actor_optim.param_groups[0]["lr"] = new_lr
                self.critic_optim.param_groups[0]["lr"] = new_lr

                
                # mini-batch updates
                np.random.shuffle(inds)
                for start in range(0, step, minibatch_size):
                    end = start + minibatch_size
                    idx = inds[start:end]
                    mini_obs = batch_obs[idx]
                    mini_acts = batch_acts[idx]
                    mini_log_prob = batch_log_probs[idx]
                    mini_advantage = A_k[idx]
                    mini_rtgs = batch_rtgs[idx]


                    # calculate pi_theta(a_t | s_t)
                    V, curr_log_probs, entropy = self.new_evaluate(mini_obs, mini_acts)

                    # calculate ratios
                    logratios = curr_log_probs - mini_log_prob
                    ratios = torch.exp(logratios)

                    # approxmiate kl divergence
                    approx_kl = ((ratios - 1) - logratios).mean()

                    # calculate surrogate losses
                    surr1 = ratios * mini_advantage
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * mini_advantage

                    # calculate actor and critic loss
                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    if V.dim() == 0:  # V is a scalar
                        V = V.unsqueeze(0)
                    critic_loss = nn.MSELoss()(V, mini_rtgs)


                    # Entropy regularisation
                    entropy_loss = entropy.mean()
                    # Discount entropy loss by given coefficient
                    actor_loss = actor_loss - self.ent_coef * entropy_loss

                    # calculate gradients and perform backpropagation for actor network
                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)  # we need to add retain_graph=True for the first network we backpropagate on or we'll get an error
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)  # Gradient clipping
                    self.actor_optim.step()

                    # calculate gradients and perform backpropagation for critic network
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)  # Gradient clipping
                    self.critic_optim.step()
                
                # If kl above threshold 
                if approx_kl > self.target_kl:
                    break

        return summed_rewards_list, timestep_counter
    

    def calculate_gae(self, rewards, values, dones):
        batch_advantages = []
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []
            last_advantage = 0

            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1 - ep_dones[t+1]) - ep_vals[t]
                else:
                    delta = ep_rews[t] - ep_vals[t]

                advantage = delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage
                advantages.insert(0, advantage)

            batch_advantages.extend(advantages)

        return torch.tensor(batch_advantages, dtype=torch.float).to(self.device)
    

    def new_evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()
        probs = self.actor(batch_obs)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs, dist.entropy()
    
     
import gym
env = gym.make("LunarLander-v2", render_mode="rgb_array")
env = RecordVideo(env=env, video_folder="./videos", name_prefix="test-video")
model = PPO(env)

summed_rewards_list, timestep_counter = model.learn(200000)
# print(f"summed rewards list {summed_rewards_list}")
print(f"summed rewards list length - > {len(summed_rewards_list)}")

# total_ep_rews = []

# for batch in summed_rewards_list:
#     for ep_rew in batch:
#         total_ep_rews.append(ep_rew)

# plt.plot(total_ep_rews)
# plt.xlabel("Number of episodes")
# plt.ylabel("Total rewards per episode")
# plt.show()


avg_returns = [np.mean(summed_rewards_list[i]) for i in range(len(summed_rewards_list))]
print("summed rewards list ->", summed_rewards_list)
print("avg returns -> ", avg_returns)

print(len(avg_returns))
print("boogeagoaoga ->>>>",timestep_counter)
print(len(timestep_counter))

plt.plot(timestep_counter, avg_returns)
plt.title('Average Episodic Return vs. Cumulative Timesteps')
plt.xlabel('Cumulative Timesteps')
plt.ylabel('Average Episodic Return')
plt.grid(True)
plt.show()