import numpy as np
from statistics import mean
import math
import heapq

def create_plan(environment, agents, timesteps, cprob):
    
    for t in range(1, timesteps):
        for a in agents:
            plan = a.plan
    		# update agent's knowledge with what it can sees
            a.knowledge.update(a.get_visible_goals())
    		# out of goals in agent's line of sight, pick goal X with highest reward
            max_goal = max(a.knowledge, key=a.knowledge.get)
            if len(a.rewards.items()) != len(a.knowledge.keys()):
                avg_other_goals = mean((value for key, value in a.rewards.items() if key not in a.knowledge.keys()))
            else:
                # all goals have already been observed!
                avg_other_goals = -float('inf')
    		# randomly choose to communicate or not with some probability
            #communicate = np.random.choice([True,False],p=[cprob, 1-cprob])
            communicate = np.random.choice([True,False],p=[0, 1])
            plan._communication_at_each_time.append(communicate)
            if communicate: 
                # determine which agent to communicate with
                ind = np.random.randint(0, len(agents))
                info_agent = agents[ind]
                # acquire other agent's knowledge
                a.knowledge.update(info_agent.knowledge)

            plan.set_knowledge(t, a.knowledge)

            # make optimal step given information  
            if a.rewards[max_goal] > avg_other_goals:
                # move toward max goal
                goal_loc = environment.goal_assignments[max_goal]
                new_location, dist = next_optimal_step(a.location, goal_loc, environment.obstacle_map)
            else:
                # move toward nearest unobserved goal
                min_dist = float('inf')
                new_location = a.location
                for key in a.rewards.keys() - a.knowledge.keys():
                    next_loc = environment.goal_assignments[key]
                    if next_loc != None:
                        ln, dist = next_optimal_step(a.location, next_loc, environment.obstacle_map)
                        if dist < min_dist:
                            min_dist = dist
                            new_location = ln

            a.location = new_location
            plan.set_location(t, new_location)

    agent_plans = {}
    for i in range(len(agents)):
        agent_plans[i] = agents[i].plan

    return agent_plans


def next_optimal_step(start_loc, goal, occupancy_map): # returns a tuple: next optimal location, distance of shortest path to goal

     def get_neighbors(node):
         neighbors = []
         for displacement in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
             xn = node[0] + displacement[0]
             yn = node[1] + displacement[1]
             if 0 <= xn < occupancy_map.shape[0] and 0 <= yn < occupancy_map.shape[1] and occupancy_map[xn, yn] == 0:
                 neighbors.append((node[0] + displacement[0], node[1] + displacement[1]))
         return neighbors

     def heuristic(node_a, node_b):
        (xa, ya) = node_a
        (xb, yb) = node_b
        return abs(xa-xb) + abs(ya-yb)

     #A*
     queue = []
     heapq.heappush(queue, (0, start_loc))
     parent = {}
     cost = {}
     parent[start_loc] = None
     cost[start_loc] = 0
     while queue:
         (c, node) = heapq.heappop(queue)
         if node == goal:
             dist = 0
             if parent[node] == None:
                return node, dist
             while parent[node] != start_loc:
                dist += 1
                node = parent[node]
             return node, dist

         for neighbor in get_neighbors(node):
             new_cost = cost[node] + 1
             if neighbor not in cost or new_cost < cost[neighbor]:
                 cost[neighbor] = new_cost
                 value = new_cost + heuristic(goal, neighbor)
                 heapq.heappush(queue, (value, neighbor))
                 parent[neighbor] = node
     return None, -1

