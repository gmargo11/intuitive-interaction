import numpy as np

def create_plan(environment, agents, timesteps, cprob):
    
    # maps agent number (index in agents list) to agent plan
    agent_plans = {}
    i = 0
    for a in agents:
    	# Initialize plan for agent at t=0
    	plan = Plan(a)
    	plan.location_at_each_time.append(a.location)
    	plan.communication_at_each_time.append(False)
    	for t in range(timesteps):
    		# how do we plan to create this path?

    		# see what is in the agent's line of sight (based on environment)
    		a.visible_goals = # some dictionary mapping goal to reward
    		# out of goals in agent's line of sight, pick goal X with highest reward
    		max_goal = max(visible_goals, key=visible_goals.get)
    		# randomly choose to communicate or not with some probability
    		communicate = np.random.choice([True,False],p=[cprob, 1-cprob])
    		# if not communicate: 
    		#	take "one step" in direction of goal with highest reward in line of sight
    		#	AKA determine new coordinates for a.location
    		if communicate:
    			# determine which agent to communicate with
    			info_agent = np.random.randint(0, len(agents))
    			# get goals in other agent's line of sight
    			non_visible_goals = {}
    			for goal in info_agent.visible_goals:
    				non_visible_goals[goal] = a.rewards[goal]
    			# determine optimal goal not in a's line of sight
    			max_info_goal = max(non_visible_goals, key=non_visible_goals.get)
    			if max_goal > max_info_goal:
    				# move in direction of max_goal
                    new_location = next_optimal_step(a.location, max_goal, environment.occupancy_map)
    			else:
    				# move in direction of max_info_goal
                    new_location = next_optimal_step(a.location, max_info_goal, environment.occupancy_map)
    		# update agent and plan
    	
    	agent_plans[i] = plan
    	i += 1

    return environment, agents, agent_plans


def next_optimal_step(start_loc, goal, occupancy_map):

    def get_neighbors(node):
        neighbors = []
        for displacement in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
            xn = node[0] + displacement[0]
            yn = node[1] + displacement[1]
            if 0 <= xn < occupancy_map.shape[0] and 0 <= yn < occupancy_map.shape[1] and occupancy_map[xn, yn] == 0:
                neighbors.append([node[0] + displacement[0], node[1] + displacement[1]])
        return neighbors

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



