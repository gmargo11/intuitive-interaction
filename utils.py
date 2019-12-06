import matplotlib.pyplot as plt
import numpy as np

def display_map_state(environment, agents, t=0, fig=0):
    state = np.array(environment.obstacle_map)
    print(state)
    for goal_location in environment.goal_locations:
        state[goal_location[0], goal_location[1]] = 2
    for agent in agents:
        #state[agent.location[0], agent.location[1]] = 3
        loc = agent.plan.get_location_at_time(t)
        state[loc[0], loc[1]] = 3

    plt.imshow(state)
    plt.title("Map")

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
    plt.bar(goals, probs)
    plt.title("Inferred goals, Agent 1")
