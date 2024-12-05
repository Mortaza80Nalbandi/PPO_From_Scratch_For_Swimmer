import gymnasium as gym
import numpy
import torch
from utils import *
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
import torch.nn.functional as F


class PPO:
    def __init__(self, env, hyperparameters):
        """
            Initializes the PPO model, including hyperparameters.
            Parameters:
                env - the environment to train on.
                hyperparameters - all extra arguments passed into PPO that should be hyperparameters.
            Returns:
                None
        """

        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)
        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # Define actor network
        self.actor = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, self.env.action_space.shape[0]),
            nn.Tanh()
        )

        # Define critic network
        self.critic = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,  # timesteps so far
            'i_so_far': 0,  # iterations so far
            'batch_lens': [],  # episodic lengths in batch
            'batch_rews': [],  # episodic returns in batch
            'actor_losses': [],  # losses of actor network in current iteration
            'critic_losses': [],  # losses of actor network in current iteration
            'avg_iter_rewards': [],  # losses of actor network in current iteration
            'actor_iter_loss': [],  # losses of actor network in current iteration
        }

    def learn_Clip(self, total_timesteps):
        """
            Train the actor and critic networks with clipping. Here is where the main PPO_CLip algorithm resides.
            Parameters:
                total_timesteps - the total number of timesteps to train for
            Return:
                None
        """
        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations ran so far
        while t_so_far < total_timesteps:  # ALG STEP 2
            # use rollout to sample data from environment
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()  # ALG STEP 3
            t_so_far += np.sum(batch_lens)
            i_so_far += 1
            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # Calculate advantage at k-th iteration
            V, _,_ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # update network loop
            for _ in range(self.n_updates_per_iteration):
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs,_ = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses.
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # update actor
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # update critic
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Log actor loss
                self.logger['actor_losses'].append(actor_loss.detach())
                self.logger['critic_losses'].append(critic_loss.detach())

            # Print a summary of our training so far
            self._log_summary()
            # Save our model if it's time
            if i_so_far % self.save_interval == 0:
                torch.save(self.actor.state_dict(), './ppo_actor' + str(i_so_far) + '.pth')
                torch.save(self.critic.state_dict(), './ppo_critic' + str(i_so_far) + '.pth')
        self.plot_training()

    def learn_adaptive_KL(self, total_timesteps):
        """
            Train the actor and critic networks. Here is where the main PPO algorithm resides.
            Parameters:
                total_timesteps - the total number of timesteps to train for
            Return:
                None
        """
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations ran so far
        while t_so_far < total_timesteps:  # ALG STEP 2
            # Autobots, roll out (just kidding, we're collecting our batch simulations here)
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()  # ALG STEP 3

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # Calculate advantage at k-th iteration
            V, _,_ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()  # ALG STEP 5

            # One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
            # isn't theoretically necessary, but in practice it decreases the variance of
            # our advantages and makes convergence much more stable and faster. I added this because
            # solving some environments was too unstable without it.
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            _, old_log_probs,mean_old = self.evaluate(batch_obs, batch_acts)
            dist_old = MultivariateNormal(mean_old.detach(), self.cov_mat)
            # This is the loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):  # ALG STEP 6 & 7
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs,curr_mean = self.evaluate(batch_obs, batch_acts)
                curr_dist = MultivariateNormal(curr_mean, self.cov_mat)
                # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                # NOTE: we just subtract the logs, which is the same as
                # dividing the values and then canceling the log with e^log.
                # For why we use log probabilities instead of actual probabilities,
                # here's a great explanation:
                # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
                # TL;DR makes gradient ascent easier behind the scenes.
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses.
                surr1 = ratios * A_k
                surr2 = torch.distributions.kl_divergence(curr_dist, dist_old)
                # Calculate actor and critic losses.
                # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                # the performance function, but Adam minimizes the loss. So minimizing the negative
                # performance function maximizes it.
                actor_loss = -torch.mean(surr1 - self.beta * surr2)
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Log actor and critic loss
                self.logger['actor_losses'].append(actor_loss.detach())
                self.logger['critic_losses'].append(critic_loss.detach())
            # KL-Divergence update
            _, _, curr_mean = self.evaluate(batch_obs, batch_acts)
            curr_dist = MultivariateNormal(curr_mean.detach(), self.cov_mat)
            DL = torch.mean(torch.distributions.kl_divergence(curr_dist, dist_old))
            if DL > 1.5 * self.delta:
                self.beta =min(self.beta *2,16)
            if DL < self.delta / 1.5:
                self.beta =max(self.beta *0.5,0.0001)




            # Print a summary of our training so far
            self._log_summary()

            # Save our model if it's time
            if i_so_far % self.save_interval == 0:
                torch.save(self.actor.state_dict(), './ppo_actor'+str(i_so_far)+'.pth')
                torch.save(self.critic.state_dict(), './ppo_critic'+str(i_so_far)+'.pth')
        self.plot_training()

    def rollout(self):
        """
            we'll need to collect a fresh batch of data each time we iterate the actor/critic networks.
            Parameters:
                None
            Return:
                batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
                batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
                batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
                batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
                batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
        """
        # Batch data. For more details, check function header.
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []
        ep_rews = []

        t = 0  # Keeps track of how many timesteps we've run so far this batch

        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < self.timesteps_per_batch:
            ep_rews = []  # rewards collected per episode

            # Reset the environment. sNote that obs is short for observation.
            obs = self.env.reset()[0]

            done = False

            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(self.max_timesteps_per_episode):
                # If render is specified, render the environment

                t += 1  # Increment timesteps ran this batch so far

                # Track observations in this batch
                batch_obs.append(obs)

                # Calculate action and make a step in the env.
                # Note that rew is short for reward.
                action, log_prob = self.get_action(torch.Tensor(obs))
                obs, rew, done, truncation, _ = self.env.step(action)
                # Track recent reward, action, and action log probability
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                # If the environment tells us the episode is terminated, break
                if done:
                    break

            # Track episodic lengths and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(numpy.array(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(numpy.array(batch_acts), dtype=torch.float)
        batch_log_probs = torch.tensor(numpy.array(batch_log_probs), dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)  # ALG STEP 4

        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):
        """
            Compute the Reward-To-Go of each timestep in a batch given the rewards.

            Parameters:
                batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

            Return:
                batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
        """
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0  # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def get_action(self, obs):
        """
            Queries an action from the actor network, should be called from rollout.

            Parameters:
                obs - the observation at the current timestep

            Return:
                action - the action to take, as a numpy array
                log_prob - the log probability of the selected action in the distribution
        """
        # Query the actor network for a mean action
        mean = self.actor(obs)

        # Create a distribution with the mean action and std from the covariance matrix above.
        # For more information on how this distribution works, check out Andrew Ng's lecture on it:
        # https://www.youtube.com/watch?v=JjB58InuTqM
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution
        action = dist.sample()

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log probability of that action in our distribution
        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        """
            Estimate the values of each observation, and the log probs of
            each action in the most recent batch with the most recent
            iteration of the actor network. Should be called from learn.

            Parameters:
                batch_obs - the observations from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of observation)
                batch_acts - the actions from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of action)

            Return:
                V - the predicted values of batch_obs
                log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
        """
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs,mean

    def _init_hyperparameters(self, hyperparameters):
        """
            Initialize default and custom values for hyperparameters

            Parameters:
                hyperparameters - the extra arguments included when creating the PPO model, should only include
                                    hyperparameters defined below with custom values.

            Return:
                None
        """
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.timesteps_per_batch = hyperparameters["timesteps_per_batch"]  # Number of timesteps to run per batch
        self.max_timesteps_per_episode = hyperparameters[
            "max_timesteps_per_episode"]  # Max number of timesteps per episode
        self.n_updates_per_iteration = hyperparameters[
            "n_updates_per_iteration"]  # Number of times to update actor/critic per iteration
        self.lr = hyperparameters["lr"]  # Learning rate of actor optimizer
        self.gamma = hyperparameters["gamma"]  # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = hyperparameters["clip"]  # Recommended 0.2, helps define the threshold to clip the ratio during SGA
        self.save_interval = hyperparameters["save_interval"]
        self.beta = hyperparameters["beta"]
        self.delta = hyperparameters["delta"]

    def _log_summary(self):
        """
            Print to stdout what we've logged so far in the most recent batch.

            Parameters:
                None

            Return:
                None
        """
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_rews = round(avg_ep_rews, 2)
        avg_actor_loss = round(avg_actor_loss, 5)
        self.logger['avg_iter_rewards'].append(avg_ep_rews)
        self.logger['actor_iter_loss'].append(avg_actor_loss)
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(avg_ep_rews)
        avg_actor_loss = str(avg_actor_loss)

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"beta  :{self.beta} ", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []

    def plot_training(self):
        # Calculate the Simple Moving Average (SMA) with a window size of 50
        print(self.logger['avg_iter_rewards'])
        sma = np.convolve(np.array(self.logger['avg_iter_rewards'], dtype=float), np.ones(50) / 50, mode='valid')

        plt.figure()
        plt.title("Rewards")
        plt.plot(self.logger['avg_iter_rewards'], label='Raw Reward', color='#F6CE3B', alpha=1)
        plt.plot(sma, label='SMA 50', color='#385DAA')
        plt.xlabel("Iters")
        plt.ylabel("Rewards")
        plt.legend()
        plt.savefig('./reward_plot.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        plt.clf()
        plt.close()

        plt.figure()
        plt.title("Loss")
        plt.plot(self.logger['actor_iter_loss'], label='Loss', color='#CB291A', alpha=1)
        plt.xlabel("Iters")
        plt.ylabel("Loss")
        plt.savefig('./Loss_plot.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    def save(self, path):
        """
        Save the parameters of the main network to a file with .pth extention.
        """
        torch.save(self.actor.state_dict(), path)

    def load(self, path):
        """
        Save the parameters of the main network to a file with .pth extention.
        """
        self.actor.load_state_dict(torch.load(path))
        self.actor.eval()


class AgentManager():
    def __init__(self, RL_hyperparams):
        self.env = gym.make('Swimmer-v4', render_mode="human" if render else None)
        self.loadPath = RL_hyperparams["RL_load_path"]
        self.agent = PPO(self.env, RL_hyperparams)
        self.RL_hyperparams = RL_hyperparams

    def train(self):
        print(f"Training", flush=True)

        # Train the PPO model with a specified total timesteps
        # NOTE: You can change the total timesteps here, I put a big number just because
        # you can kill the process whenever you feel like PPO is converging
        if self.RL_hyperparams["PPO_Clip"]:
            self.agent.learn_Clip(total_timesteps=RL_hyperparams["training_timesteps"])
        elif self.RL_hyperparams["PPO_adaptive_KL"]:
            self.agent.learn_adaptive_KL(total_timesteps=RL_hyperparams["training_timesteps"])

    def test(self, max_episodes):
        self.agent.load(self.loadPath)
        for i in range(max_episodes):
            rsum = 0
            ob = self.env.reset()
            # Check if it is a testing episode
            j = 0
            new_obs = ob[0]
            while True:
                j += 1
                obs = new_obs
                action, _ = self.agent.get_action(torch.Tensor(obs))
                new_obs, reward, done, truncation, _ = self.env.step(action)
                rsum += reward
                if (j >= RL_hyperparams["max_timesteps_per_episode"]):
                    done = True
                # If episode is done, evaluate the results of this episode and start a new episode
                if done:
                    print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                    break

        self.env.close()


def TESTeNV():
    render = True
    env = gym.make('Swimmer-v4', render_mode="human" if render else None)
    env.reset()
    for i in range(3000):
        actopn = env.action_space.sample()
        env.step(actopn)
        print(actopn)
    env.reset()


if __name__ == '__main__':
    # Parameters:
    train_mode = False
    render = not train_mode
    clip = False
    RL_hyperparams = {
        "train_mode": train_mode,
        "RL_load_path": f'./clip0.2/weights/ppo_actor'+'1950'+'.pth',
        "save_interval":50,
        "PPO_Clip": clip,
        "PPO_adaptive_KL": not clip,
        'timesteps_per_batch': 2048,
        'max_timesteps_per_episode': 2048,
        'gamma': 0.999,
        'n_updates_per_iteration': 5,
        'training_timesteps': 1_000_000,
        'lr': 3e-4,
        'clip': 0.003,
        'beta': 1,
        'delta': 0.0005,
        'render': True,
        'render_every_i': 10
    }

    # Run
    PPO_Agent = AgentManager(RL_hyperparams)  # Define the instance
    # Train
    if train_mode:
        PPO_Agent.train()
    else:
        # Test
        PPO_Agent.test(max_episodes=10)
        pass
