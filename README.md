# drlgeb
A sample DRL lib use tf.keras and gym.
Just writing ...

Now just implement the batch a3c, other algorithms will be updated in the future.

## Requirement
- Linux/windows/osx
- python3.8
- tensorflow2.4
- gym[atari]
- numpy

## Installation
```shell script
pip install drlgeb
```

## Get start

#### A3C
```python
from drlgeb.ac import A3C



env_id = "SpaceInvaders-v0"
agent = A3C(env_id=env_id)

# train
agent.learn()

# test
model_path = "..."
agent.play(episodes=5, model_path=model_path)

```
