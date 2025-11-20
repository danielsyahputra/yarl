import numpy as np
from helper.environment import BaseEnvironment

class Environment(BaseEnvironment):
    def __init__(self):
        reward = None
        observation = None
        termination = None
        self.reward_obs_term = (reward, observation, termination)
        self.count = 0
        self.arms = []
        self.seed = None

    def env_init(self, env_info={}):
        num_of_arms = env_info.get('n_arms', 10)
        self.arms = np.random.randn(num_of_arms)
        local_observation = 0
        self.reward_obs_term = (0.0, local_observation, False)

    def env_start(self):
        return self.reward_obs_term[1]
    
    def env_step(self, action):
        reward = self.arms[action] + np.random.randn()
        obs = self.reward_obs_term[1]

        self.reward_obs_term = (reward, obs, False)

        return self.reward_obs_term
    
    def env_cleanup(self):
        pass

    def env_message(self, message):
        if message == "what is the current reward?":
            return "{}".format(self.reward_obs_term[0])

        return "I don't know how to respond to your message"
