import os
import yaml
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

class RewardShaper(gym.RewardWrapper):
  """Reward shaper that encourages forward movement and penalizes staying still."""
  def __init__(self, env):
    super().__init__(env)
    self.prev_x = 0.0
    self.step_penalty = -0.05

  def reward(self, r):
    current_x = self.unwrapped.hull.position.x
    # Bonus scales with how far agent moves forward each step
    forward_bonus = (current_x - self.prev_x) * 2.0
    self.prev_x = current_x
    return r + forward_bonus + self.step_penalty

  def reset(self, **kwargs):
    obs, info = self.env.reset(**kwargs)
    self.prev_x = self.unwrapped.hull.position.x
    return obs, info

def load_config(config_path: str = "config.yaml") -> dict:
  with open(config_path, "r") as f:
    return yaml.safe_load(f)

def make_env(env_name: str, render_mode: str = None) -> gym.Env:
  env = gym.make(env_name, render_mode=render_mode)
  env = RewardShaper(env)
  env = Monitor(env)
  return env

def create_agent(env: gym.Env, config: dict) -> SAC:
  """
  Create a SAC agent with config parameters.

  SAC (Soft Actor-Critic) key components:
  - Actor: Policy network that outputs actions
  - Critic: Two Q-networks that estimate action values
  - Entropy: Encourages exploration by rewarding randomness
  """
  agent = SAC(
      policy="MlpPolicy",
      env=env,
      learning_rate=config["learning_rate"],
      buffer_size=config["buffer_size"],
      batch_size=config["batch_size"],
      gamma=config["gamma"],
      tau=config["tau"],
      ent_coef=config["ent_coef"],
      learning_starts=config["learning_starts"],
      tensorboard_log="logs",
      verbose=0,
  )
  return agent

def get_callbacks(config: dict, env: gym.Env) -> list:
  os.makedirs("models", exist_ok=True)

  # Save model checkpoints periodically for resuming interrupted training
  checkpoint_cb = CheckpointCallback(
      save_freq=config["save_freq"],
      save_path="models",
      name_prefix="sac_bipedal",
  )

  # Checks whether to save the best model based on evaluation reward
  eval_env = make_env(config["env_name"])
  eval_cb = EvalCallback(
      eval_env,
      best_model_save_path="models",
      log_path="logs",
      eval_freq=config["save_freq"],
      n_eval_episodes=5,
      deterministic=True,
  )

  # Print episode stats to console during training
  log_cb = EpisodeLogger(log_interval=config["log_interval"])
  return [checkpoint_cb, eval_cb, log_cb]

class EpisodeLogger(BaseCallback):
  def __init__(self, log_interval: int = 10):
    super().__init__()
    self.log_interval = log_interval
    self.episode_count = 0
    self.current_episode_reward = 0.0
    self.current_episode_steps = 0

  def _on_step(self) -> bool:
    step_reward = float(self.locals["rewards"][0])
    episode_finished = self.locals["dones"][0]
    self.current_episode_reward += step_reward
    self.current_episode_steps += 1

    if episode_finished:
      self.episode_count += 1
      if self.episode_count % self.log_interval == 0:
        print(
            f"Episode {self.episode_count:4d} | "
            f"Reward {self.current_episode_reward:8.2f} | "
            f"Steps {self.current_episode_steps:4d}"
        )
      self.current_episode_reward = 0.0
      self.current_episode_steps = 0

    return True

def load_agent(model_path: str, env: gym.Env = None) -> SAC:
  return SAC.load(model_path, env=env)
