import os
import glob
from sac import load_config, make_env, create_agent, get_callbacks, load_agent

def train():
  config = load_config()
  print(f"Training SAC on {config['env_name']}")
  print(f"Total timesteps: {config['total_timesteps']}")

  env = make_env(config["env_name"])

  # Check for existing checkpoints
  checkpoint_path = None
  if os.path.exists("models"):
    # Look for checkpoint files
    checkpoint_files = glob.glob("models/sac_bipedal_*_steps.zip")
    if checkpoint_files:
      # Get the most recent checkpoint, sorted by the number of steps
      checkpoint_files.sort(key=lambda x: int(x.split('_')[-2]))
      checkpoint_path = checkpoint_files[-1]
      print(f"Found checkpoint: {checkpoint_path}")

  if checkpoint_path:
    # Load existing model from checkpoint
    agent = load_agent(checkpoint_path, env=env)
    # Extract timesteps from checkpoint filename
    checkpoint_steps = int(checkpoint_path.split('_')[-2])
    remaining_steps = config["total_timesteps"] - checkpoint_steps
    print(f"Resuming training from checkpoint at {checkpoint_steps} steps")
  else:
    # Create new agent from scratch
    agent = create_agent(env, config)
    remaining_steps = config["total_timesteps"]
    print("Starting training from scratch")

  callbacks = get_callbacks(config, env)

  print("For tensorboard: tensorboard --logdir logs")
  print("-" * 50)

  agent.learn(
      total_timesteps=remaining_steps,
      callback=callbacks,
      log_interval=config["log_interval"],
      progress_bar=True,
      reset_num_timesteps=False,
  )

  env.close()

if __name__ == "__main__":
  train()
