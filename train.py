from sac import load_config, make_env, create_agent, get_callbacks

def train():
  config = load_config()
  print(f"Training SAC on {config['env_name']}")
  print(f"Total timesteps: {config['total_timesteps']}")

  env = make_env(config["env_name"])
  agent = create_agent(env, config)

  callbacks = get_callbacks(config, env)

  print("For tensorboard: tensorboard --logdir logs")
  print("-" * 50)

  agent.learn(
      total_timesteps=config["total_timesteps"],
      callback=callbacks,
      log_interval=config["log_interval"],
      progress_bar=True,
  )

  env.close()

if __name__ == "__main__":
  train()
