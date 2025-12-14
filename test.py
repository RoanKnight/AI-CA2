import argparse
import numpy as np
from sac import load_config, make_env, load_agent

def test(model_path: str = None, episodes: int = None):
  config = load_config()

  episodes = episodes or config["eval_episodes"]
  model_path = model_path or "models/best_model"

  print(f"Testing agent from: {model_path}")

  env = make_env(config["env_name"], render_mode="human")

  agent = load_agent(model_path, env=env)

  for ep in range(episodes):
    obs, _ = env.reset()
    episode_finished = False
    episode_reward = 0
    episode_steps = 0

    while not episode_finished:
      action, _ = agent.predict(obs, deterministic=True)
      obs, step_reward, terminated, truncated, _ = env.step(action)
      episode_finished = terminated or truncated
      episode_reward += step_reward
      episode_steps += 1

    print(f"Episode {ep + 1}: Reward = {episode_reward:.2f}, Steps = {episode_steps}")

  env.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Test trained SAC agent")
  parser.add_argument("--model", type=str, help="Path to model file")
  parser.add_argument("--episodes", type=int, help="Number of test episodes")

  args = parser.parse_args()
  test(
      model_path=args.model,
      episodes=args.episodes,
  )
