from environment import Environment
from agent import Agent
from plan import Plan
from planning import create_plan
from inference import infer_communication, infer_goal
from utils import display_map_state, display_belief_state, display_inferred_goals

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



def two_goals_example():
    # define environment
    obstacle_map = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    goal_locations = [(3, 12), (11, 12)]
    env = Environment(obstacle_map=obstacle_map, goal_locations=goal_locations)


    # define agents
    agent1 = Agent(rewards={'A': 3, 'B': 10}, initial_location=(3, 7), initial_beliefs=np.array([[0.5, 0.5], [0.5, 0.5]]), environment=env)
    agent2 = Agent(rewards={'A': 5, 'B': 8}, initial_location=(11, 7), initial_beliefs=np.array([[0.5, 0.5], [0.5, 0.5]]), environment=env)

    # generate plans
    plan = create_plan(environment=env, agents=[agent1, agent2], timesteps=30, cprob=0.2)
    for p in plan:
        print(plan[p]._location_at_each_time)

    plt.figure()
    for t in range(agent1.plan.get_duration()):
         display_map_state(environment=env, agents=[agent1, agent2], t=t)
         plt.waitforbuttonpress()
    plt.show()

    # infer communication
    #communication_prob_at_each_time = infer_communication(environment=env, agents=[agent1, agent2], plan=plan)

def three_goals_example():
    # define environment
    obstacle_map = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

    goal_locations = [(3, 12), (7, 12), (12, 12)]
    goals = ['A','B','C','D','E']
    goal_assignments = {'A': goal_locations[0], 'B': goal_locations[1], 'C': goal_locations[2], 'D': None, 'E': None}
    env = Environment(obstacle_map=obstacle_map, goal_locations=goal_locations, goal_assignments=goal_assignments)


    # define agents
    agent1 = Agent(rewards={'A': 3, 'B': 5, 'C': 10, 'D': 5, 'E': 5}, initial_location=(7, 7), initial_beliefs=np.array([[0.5, 0.5], [0.5, 0.5]]), environment=env)
    agent2 = Agent(rewards={'A': 5, 'B': 8, 'C': 7, 'D': 5, 'E': 5}, initial_location=(11, 7), initial_beliefs=np.array([[0.5, 0.5], [0.5, 0.5]]), environment=env)

    #agent1.plan.set_next_location((4, 7))
    #agent2.plan.set_next_location((10, 7))

    # generate plans
    plan = create_plan(environment=env, agents=[agent1, agent2], timesteps=30, cprob=0.0)
    for p in plan:
        print(plan[p]._location_at_each_time)

    fig = plt.figure()
    maps = []
    inferences = []
    for t in range(agent1.plan.get_duration()):
         #plt.figure()
         plt.cla()
         plt.subplot(1, 2, 1)
         display_map_state(environment=env, agents=[agent1, agent2], t=t)
         plt.subplot(1, 2, 2)
         display_inferred_goals(infer_goal(agent1, t))
         plt.waitforbuttonpress()
    

    # infer communication
    #communication_prob_at_each_time = infer_communication(environment=env, agents=[agent1, agent2], plan=plan)
    # generate GIF
    fig2 = plt.figure()
    maps = []
    for t in range(agent1.plan.get_duration()):
         maps.append([display_map_state(environment=env, agents=[agent1, agent2], t=t)])
    print(maps)
    ani = animation.ArtistAnimation(fig2, maps, interval=200, blit=True,
                                repeat_delay=1000)
    ani.save('scenario2.gif', writer='pillow')


if __name__ == "__main__":
    three_goals_example()