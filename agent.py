class Agent:
    def __init__(self, rewards: dict, initial_location: tuple, initial_beliefs: list, environment):
        # rewards is dictionary mapping goal: reward
        self.rewards = rewards
        self.visible_goals = {}
        self.location = initial_location
        self.initial_beliefs = initial_beliefs
        self.environment = environment
        #self.location_at_each_time = []
        #self.communication_at_each_time = []
        #self.beliefs_at_each_time = []
        #self.rewards_at_each_time = []