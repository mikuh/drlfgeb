from drlgeb.ac import A3C


env_id = "SpaceInvaders-v0"
agent = A3C(env_id=env_id)

# train
agent.learn()

# test
model_path = "..."
agent.play(episodes=5, model_path=model_path)
