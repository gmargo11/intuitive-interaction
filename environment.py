import math

class Environment:
    def __init__(self, obstacle_map: list, obstacle_endpoints: list, goal_locations: list):
        self.obstacle_map = obstacle_map
        self.obstacle_endpoints = obstacle_endpoints
        self.goal_locations = goal_locations

    def is_visible(self, ax, ay, bx, by):
        steps = max(abs(bx-ax), abs(by-ay)) - 1
        dx = (bx-ax)/steps
        dy = (by-ay)/steps
        
        x = ax + 0.5*dx
        y = ay + 0.5*dy

        points = []
        for i in range(steps):
            if abs(bx-ax)>abs(by-ay):
                points.append([int(round(y)), int(math.floor(x))])
                points.append([int(round(y)), int(math.ceil(x))])
            else:
                points.append([int(math.floor(y)), int(round(x))])
                points.append([int(math.ceil(y)), int(round(x))])
            x += dx
            y += dy

        for point in points:
            if self.obstacle_map[point[1], point[0]] != 0:
                return False
        return True