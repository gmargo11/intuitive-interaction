class Environment:
    def __init__(self, obstacle_map: list, goal_locations: list):
        self.obstacle_map = obstacle_map
        self.goal_locations = goal_locations