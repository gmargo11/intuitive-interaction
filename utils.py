import matplotlib.pyplot as plt

def display_state(environment, agents):
    state = environment.obstacle_map[:]
    print(state)
    for goal_location in environment.goal_locations:
        state[goal_location[0], goal_location[1]] = 2
    for agent in agents:
        state[agent.location[0], agent.location[1]] = 3

    plt.figure()
    plt.imshow(state)
    plt.show()