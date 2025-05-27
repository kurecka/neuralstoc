import functools

from brax import envs
from brax.io import model
from neuralstoc.rl import train_sac


class SAC:
    """
    Soft Actor-Critic (SAC) implementation for NeuralStoc.
    
    This class provides an interface to train neural controllers using the SAC algorithm
    before they are refined with certificates in the learner-verifier loop. SAC is an
    off-policy actor-critic method that uses entropy regularization for exploration.
    
    Attributes:
        env_name: Name of the environment to train on
        env_dim: Dimension of the environment state space (if specified)
        hidden_dim: Size of hidden layers in the neural networks
    """
    
    def __init__(self, env_name, env_dim=None, p_hidden=[256, 256], num_timesteps=1_200_000_000):
        """
        Initialize the SAC trainer.
        
        Args:
            env_name: Name of the environment to train on
            env_dim: Dimension of the environment state space (if specified)
            p_hidden: Size of hidden layers in the neural networks
        """
        self.env_name = env_name
        self.p_hidden = p_hidden

        if env_dim is None:
            self.env = envs.get_environment(env_name=env_name)
        else:
            self.env = envs.get_environment(env_name=env_name, num_dims=env_dim)
        self.w_decay = 20.0
        self.train_fn = functools.partial(train_sac.train, num_timesteps=num_timesteps, num_evals=100, reward_scaling=0.1,
                                          episode_length=self.env.episode_length, normalize_observations=True, action_repeat=1,
                                          discounting=0.99, learning_rate=1e-6, num_envs=2048,
                                          batch_size=1024, seed=1, grad_updates_per_step=1, max_devices_per_host=1,
                                          max_replay_size=1048576, min_replay_size=8192, weight_decay=self.w_decay)


    @staticmethod
    def progress(num_steps, metrics):
        print(f'reward step {num_steps}: {metrics["eval/episode_reward"]}')


    def train(self, filename):
        """
        Train a policy using the SAC algorithm.
        
        This method configures and runs the SAC training process, and saves the
        resulting policy model to a file if specified.
        
        Args:
            filename: Path to save the trained model (default: None)
            
        Returns:
            dict: The trained policy parameters
        """
        make_inference_fn, params, _ = self.train_fn(environment=self.env, progress_fn=self.progress, p_hidden=self.p_hidden)
        model.save_params(filename, params)
        return params
    
    @staticmethod
    def dummy_progress(num_steps, metrics):
        pass
    
    def dummy_obs_norm(self):
        print("Computing dummy obs norm")
        strain_fn = functools.partial(train_sac.train, num_timesteps=8193, num_evals=100, reward_scaling=0.1,
                                          episode_length=self.env.episode_length, normalize_observations=True, action_repeat=1,
                                          discounting=0.99, learning_rate=1e-6, num_envs=2048,
                                          batch_size=1024, seed=1, grad_updates_per_step=1, max_devices_per_host=1,
                                          max_replay_size=1048576, min_replay_size=8192, weight_decay=self.w_decay)
        _, params, _ = strain_fn(environment=self.env, progress_fn=self.dummy_progress, p_hidden=self.p_hidden)
        return params[0]


