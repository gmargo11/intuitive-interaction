class Environment:
    def __init__(self, obstacle_map: list, obstacle_endpoints: list, goal_locations: list):
        self.obstacle_map = obstacle_map
        self.obstacle_endpoints = obstacle_endpoints
        self.goal_locations = goal_locations