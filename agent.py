from plan import Plan

class Agent:
    def __init__(self, name: str, rewards: dict, initial_location: tuple, initial_beliefs: list, environment):
        self.rewards = rewards
        self.location = initial_location
        self.initial_beliefs = initial_beliefs
        self.environment = environment
        self.plan = Plan(initial_location=initial_location)
        self.knowledge = {}
        self.name = name

    def get_visible_goals(self, loc=None):
        # return dictionary of visible goals in agent's line of sight
        # goal: reward
        if loc==None: loc = self.location

        visible_goals = {}
        for goal in self.environment.goal_locations:
            if self.environment.is_visible(loc[0], loc[1], goal[0], goal[1]):
                visible_goals[goal] = self.rewards[goal]
        return visible_goals