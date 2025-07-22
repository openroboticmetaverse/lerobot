import os

from gymnasium.envs.registration import register

__version__ = "0.0.1"

ASSETS_PATH = os.path.dirname(__file__)

register(
    id="LerobotCube-v0",
    entry_point="lerobot.sim.lerobot_env:LerobotEnv",
    max_episode_steps=50,
)