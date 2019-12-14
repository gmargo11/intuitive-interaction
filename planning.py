import numpy as np
from statistics import mean
import math
import heapq

def create_plan(environment, agents, timesteps, cprob=0.0, ctime=-1):
    
    for t in range(1, timesteps):
        for ai in range(len(agents)):
            a = agents[ai]
            plan = a.plan
    		# update agent's knowledge with what it can sees
            a.knowledge.update(a.get_visible_goals())
    		# randomly choose to communicate or not with some probability
            communicate = np.random.choice([True,False],p=[cprob, 1-cprob])
            #communicate = np.random.choice([True,False],p=[0, 1])
            plan._communication_at_each_time.append(communicate)
            if communicate or t==ctime: 
                # determine which agent to communicate with
                info_agent = np.random.choice(agents[:ai] + agents[ai+1:])
                # acquire other agent's knowledge
                a.knowledge.update(info_agent.knowledge)

            new_location, dist = next_step_given_beliefs_rewards(a.location, a.knowledge, a.rewards, environment)
            
            if dist > 0:
                a.location = new_location
                plan.set_location(t, a.location)
                plan.set_knowledge(t, a.knowledge.copy())


    agent_plans = {}
    for i in range(len(agents)):
        agent_plans[i] = agents[i].plan

    return agent_plans


def next_step_given_beliefs_rewards(start_loc, beliefs, rewards, environment):
    # out of goals in agent's knowledge, pick goal X with highest reward
    max_goal = max(beliefs, key=beliefs.get)
    if len(rewards.items()) != len(beliefs.keys()):
        avg_other_goals = mean((value for key, value in rewards.items() if key not in beliefs.keys()))
    else:
        # all goals have already been observed!
        avg_other_goals = -float('inf')

    # make optimal step given information  
    if rewards[max_goal] > avg_other_goals:
        # move toward max goal
        goal_loc = environment.goal_assignments[max_goal]
        new_location, dist = next_optimal_step(start_loc, goal_loc, environment.obstacle_map)
    else:
        # move toward nearest unobserved goal
        dist = float('inf')
        new_location = start_loc
        for key in rewards.keys() - beliefs.keys():
            next_loc = environment.goal_assignments[key]
            if next_loc != None:
                ln, ldist = next_optimal_step(start_loc, next_loc, environment.obstacle_map)
                if ldist < dist:
                    dist = ldist
                    new_location = ln

    return new_location, dist


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

