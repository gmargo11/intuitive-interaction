import numpy as np
from statistics import mean
import math
import heapq

def create_plan(environment, agents, timesteps, cprob):
    
    for t in range(1, timesteps):
        for a in agents:
            plan = a.plan
    		# update agent's knowledge with what it can see
            a.knowledge.update(a.get_visible_goals())
    		# out of goals in agent's line of sight, pick goal X with highest reward
            max_goal = max(a.knowledge, key=a.knowledge.get)
            if len(a.rewards.items()) != len(a.knowledge.keys()):
                avg_other_goals = mean((value for key, value in a.rewards.items() if key not in a.knowledge.keys()))
            else:
                # all goals have already been observed!
                avg_other_goals = -float('inf')
    		# randomly choose to communicate or not with some probability
            communicate = np.random.choice([True,False],p=[cprob, 1-cprob])
            #communicate = np.random.choice([True,False],p=[0, 1])
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
                new_location, dist = next_optimal_step(a.location, max_goal, environment.obstacle_map)
            else:
                # move toward nearest unobserved goal
                min_dist = float('inf')
                new_location = a.location
                for key in a.rewards.keys() - a.knowledge.keys():
                    ln, dist = next_optimal_step(a.location, key, environment.obstacle_map)
                    if dist < min_dist:
                        min_dist = dist
                        new_location = ln

           # if communicate:
           #     #print('comm')
    	   # 	# determine which agent to communicate with
           #     ind = np.random.randint(0, len(agents))
           #     info_agent = agents[ind]
    	#		# get goals in other agent's line of sight
        #        info_agent_knowledge = info_agent.knowledge
    	#		# determine optimal goal not in a's line of sight
        #        max_info_goal = max(info_agent_knowledge, key=info_agent_knowledge.get)
        #        if max_goal > max_info_goal and max_goal > avg_other_goals:
    	#			# move in direction of max_goal
        #            new_location, dist = next_optimal_step(a.location, max_goal, environment.obstacle_map)
        #        elif max_info_goal > avg_other_goals:
    	#			# move in direction of max_info_goal
        #            new_location, dist = next_optimal_step(a.location, max_info_goal, environment.obstacle_map)
        #        else:
        #            # move in direction of nearest unobserved goal
        #            min_dist = float('inf')
        #            new_location = a.location
        #            for key in a.rewards.keys() - [max_goal]:
        #                ln, dist = next_optimal_step(a.location, key, environment.obstacle_map)
        #                if dist < min_dist:
        #                    min_dist = dist
        #                    new_location = ln

            a.location = new_location
            plan.set_location(t, new_location)

    agent_plans = {}
    for i in range(len(agents)):
        agent_plans[i] = agents[i].plan

    return agent_plans


#def get_closest_obstacle(agent, environment):
#    # (distance, coordinates)
#    closest_obstacle = (float('inf'), None)
#    for obstacle in environment.obstacle_endpoints:
#        for endpoint in obstacle:
#            dist = ((agent.location[0]-endpoint[0])**2+(agent.location[1]-endpoint[1])**2)**(1/2)
#            if dist < closest_obstacle[0]:
#                closest_obstacle = (dist, endpoint)
#    return closest_obstacle[1]



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

    
    #ax, ay = start_loc[0], start_loc[1]
    #bx, by = goal[0], goal[1]

    #steps = max(abs(bx-ax), abs(by-ay)) - 1
    #if steps <= 0:
    #    return goal
    #dx = (bx-ax)/steps
    #dy = (by-ay)/steps
        
    #x = ax + 0.5*dx
    #y = ay + 0.5*dy

    #points = []
    #for i in range(steps):
    #    if abs(bx-ax)>abs(by-ay):
    #        points.append([int(round(y)), int(math.floor(x))])
    #        points.append([int(round(y)), int(math.ceil(x))])
    #    else:
    #        points.append([int(math.floor(y)), int(round(x))])
    #        points.append([int(math.ceil(y)), int(round(x))])
    #    x += dx
    #    y += dy

    #next_step = None
    #for pt in points:
    #    if (pt[1], pt[0]) != start_loc:
    #        next_step = (pt[1], pt[0])
    #        break

    #return next_step

