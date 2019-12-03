import matplotlib.pyplot as plt

def display_map_state(environment, agents):
    state = environment.obstacle_map[:]
    print(state)
    for goal_location in environment.goal_locations:
        state[goal_location[0], goal_location[1]] = 2
    for agent in agents:
        state[agent.location[0], agent.location[1]] = 3

    plt.figure()
    plt.imshow(state)
    plt.title("Map")

def display_belief_state(agent):
    plt.figure()
    num_rewards = len(agent.rewards)
    for i in range(num_rewards):
        plt.subplot(1, num_rewards, i+1)
        plt.bar(range(len(agent.initial_beliefs[i, :])), agent.initial_beliefs[i, :])
        plt.title("Initial Beliefs, Object %s" % i)
