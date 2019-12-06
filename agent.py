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

    def get_visible_goals(self):
        # return dictionary of visible goals in agent's line of sight
        # goal: reward

        visible_goals = {}
        for goal in self.environment.goal_assignments:
            goal_loc = self.environment.goal_assignments[goal]
            if goal_loc != None and self.environment.is_visible(self.location[0], self.location[1], goal_loc[0], goal_loc[1]):
                visible_goals[goal] = self.rewards[goal]
        return visible_goals