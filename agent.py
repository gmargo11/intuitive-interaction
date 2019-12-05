from plan import Plan

class Agent:
    def __init__(self, rewards: dict, initial_location: tuple, initial_beliefs: list, environment):
        self.rewards = rewards
        self.location = initial_location
        self.initial_beliefs = initial_beliefs
        self.environment = environment
        self.plan = Plan()
        self.knowledge = {}

    def get_visible_goals(self, environment):
        # return dictionary of visible goals in agent's line of sight
        # goal: reward

        

        raise NotImplementedError