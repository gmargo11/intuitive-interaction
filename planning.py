import numpy as np

def create_plan(environment, agents, timesteps, cprob):
    
    # maps agent number (index in agents list) to agent plan
    agent_plans = {}
    i = 0
    for a in agents:
    	# Initialize plan for agent at t=0
    	plan = a.plan
    	plan.location_at_each_time.append(a.location)
    	plan.communication_at_each_time.append(False)
    	for t in range(timesteps):

            # TO DO:
            # get_visible_goals fn

    		# update agent's knowledge with what it can see
            a.knowledge.update(a.get_visible_goals())
    		# out of goals in agent's line of sight, pick goal X with highest reward
    		max_goal = max(knowledge, key=knowledge.get)
            avg_other_goals = sum((value for key, value in a.rewards.iteritems() if key != max_goal))
    		# randomly choose to communicate or not with some probability
    		communicate = np.random.choice([True,False],p=[cprob, 1-cprob])
    		if not communicate: 
                if a.rewards[max_goal] > avg_other_goals:
                    new_location = next_optimal_step(a.location, max_goal, environment.occupancy_map)
                else:
                    # move toward closest obstacle
                    closest_obstacle = get_closest_obstacle(a, environment)
                    new_location = next_optimal_step(a.location, closest_obstacle, environment.occupancy_map)

    		if communicate:
    			# determine which agent to communicate with
    			info_agent = np.random.randint(0, len(agents))
    			# get goals in other agent's line of sight
                info_agent_knowledge = info_agent.knowledge
    			# determine optimal goal not in a's line of sight
    			max_info_goal = max(info_agent_knowledge, key=info_agent_knowledge.get)
    			if max_goal > max_info_goal and max_goal > avg_other_goals:
    				# move in direction of max_goal
                    new_location = next_optimal_step(a.location, max_goal, environment.occupancy_map)
                elif max_info_goal > avg_other_goals:
    				# move in direction of max_info_goal
                    new_location = next_optimal_step(a.location, max_info_goal, environment.occupancy_map)
                else:
                    # move toward closest obstacle
                    closest_obstacle = get_closest_obstacle(a, environment)
                    new_location = next_optimal_step(a.location, closest_obstacle, environment.occupancy_map)

            plan.location_at_each_time.append(new_location)
    		
    	agent_plans[i] = plan
    	i += 1

    return environment, agents, agent_plans


def get_closest_obstacle(agent, environment):
    # (distance, coordinates)
    closest_obstacle = (float('inf'), None)
    for obstacle in environment.obstacle_endpoints:
        for endpoint in obstacle:
            dist = ((agent.location[0]-endpoint[0])**2+(agent.location[1]-endpoint[1]**2))**(1/2)
            if dist < closest_obstacle[0]:
                closest_obstacle = (dist, endpoint)
    return closest_obstacle



def next_optimal_step(start_loc, goal, occupancy_map):

    def get_neighbors(node):
        neighbors = []
        for displacement in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
            xn = node[0] + displacement[0]
            yn = node[1] + displacement[1]
            if 0 <= xn < occupancy_map.shape[0] and 0 <= yn < occupancy_map.shape[1] and occupancy_map[xn, yn] == 0:
                neighbors.append([node[0] + displacement[0], node[1] + displacement[1]])
        return neighborsx

    # BFS
    queue = [[start_loc]]
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node == goal:
            return path[1]
        for neighbor in get_neighbors(node):
            newpath = path[:]
            newpath.append(neighbor)
            queue.append(newpath)
    return None

