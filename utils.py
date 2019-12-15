import matplotlib.pyplot as plt
import numpy as np

def display_map_state(environment, agents, t=0, fig=0):
    state = (1-np.array(environment.obstacle_map)) * 90 + np.array(environment.obstacle_map) * 0
    for goal_location in environment.goal_locations:
        state[goal_location[0], goal_location[1]] = 205
    for agent in agents:
        for ti in range(t):
            loc = agent.plan.get_location_at_time(ti)
            state[loc[0], loc[1]] = 235
        loc = agent.plan.get_location_at_time(t)
        state[loc[0], loc[1]] = 255

    plt.title("Map")
    im = plt.imshow(state, cmap='Spectral')

    for goal_location in environment.goal_locations:
        plt.text(goal_location[1]-0.25, goal_location[0]+0.3, str(environment.goal_assignments_inv[goal_location]), fontsize=8)
    for agent in agents:
        agent_loc = agent.plan.get_location_at_time(t)
        for ti in range(t):
            loc = agent.plan.get_location_at_time(ti)
            if loc != agent_loc:
                plt.text(loc[1]-0.5, loc[0]+0.5, str(ti), fontsize=8)
        plt.text(agent_loc[1]-0.5, agent_loc[0]+0.5, str(agent.name), fontsize=8)

    return im

def display_belief_state(agent):
    num_rewards = len(agent.rewards)
    for i in range(num_rewards):
        plt.subplot(1, num_rewards, i+1)
        plt.bar(range(len(agent.initial_beliefs[i, :])), agent.initial_beliefs[i, :])
        plt.title("Initial Beliefs, Object %s" % i)

def display_inferred_goals(inferred_goals, fig=1):
    goals = list(inferred_goals.keys())
    probs = [inferred_goals[goal] for goal in goals]
    goals = [str(goal) for goal in goals]
    plt.title("Inferred goals, Agent 1") 
    return plt.bar(goals, probs)
