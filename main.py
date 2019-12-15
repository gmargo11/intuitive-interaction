from environment import Environment
from agent import Agent
from plan import Plan
from planning import create_plan
from inference import infer_communication, infer_goal
from utils import display_map_state, display_belief_state, display_inferred_goals

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# two goals example for testing purposes
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


# four goals example used to construct scenarios
def four_goals_example():
    # define environment
    obstacle_map = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

    goal_locations = [(3, 12), (9, 14), (15, 14), (3, 2)]
    goals = ['A','B','C','D','E']
    goal_assignments = {'A': goal_locations[0], 'B': goal_locations[1], 'C': goal_locations[2], 'D': goal_locations[3], 'E': None}
    env = Environment(obstacle_map=obstacle_map, goal_locations=goal_locations, goal_assignments=goal_assignments)


    agent1 = Agent(name="A1", rewards={'A': 3, 'B': 5, 'C': 10, 'D': 5, 'E': 5}, initial_location=(9, 10), initial_beliefs=np.array([[0.5, 0.5], [0.5, 0.5]]), environment=env)
    agent2 = Agent(name="A2", rewards={'A': 5, 'B': 8, 'C': 7, 'D': 3, 'E': 5}, initial_location=(15, 10), initial_beliefs=np.array([[0.5, 0.5], [0.5, 0.5]]), environment=env)

    # generate plans
    plan = create_plan(environment=env, agents=[agent1, agent2], timesteps=80, cprob=0.0, ctime=-1)

    fig = plt.figure()
    vis_map_state = True    

    maps = []
    inferences = []
    communication = []
    probs = {}
    for goal_loc in env.goal_locations:
        probs[goal_loc] = []
    for t in range(agent1.plan.get_duration()):
        c = infer_communication(agents=[agent1,agent2], environment=env,t=t)
        if 'A1' in c:
           communication.append(c['A1'])
        # g = infer_goal(agent1, t)
        # for goal in g:
        #    probs[goal].append(g[goal])
    #       fig = plt.figure()
        if vis_map_state:
           fig = plt.figure()
           plt.cla()
           ax = plt.subplot(1, 2, 1)
           plt.cla()
           display_map_state(environment=env, agents=[agent1, agent2], t=t)
           ax.axis('off')
    #      ax = plt.subplot(1, 2, 2)
    #      plt.cla()
    #      ax.axis('off')
    #      display_inferred_goals(infer_goal(agent1, t))
           print(infer_communication(agents=[agent1, agent2], environment=env, t=t))
           plt.waitforbuttonpress()
    print(len(communication))
    print(communication)

    # plot goal probabilities
    # timesteps = [i for i in range(agent1.plan.get_duration())]
    # plt.plot(timesteps, probs[(3, 12)],marker='',linewidth=2, linestyle=':',label='A')
    # plt.plot(timesteps, probs[(9, 14)],marker='',linewidth=2, linestyle='-',label='B')
    # plt.plot(timesteps, probs[(15, 14)],marker='',linewidth=2, linestyle='--',label='C')
    # plt.plot(timesteps, probs[(3, 2)],marker='',linewidth=2, linestyle='-.',label='D')
    # plt.xlabel('Timestep')
    # plt.ylabel('Probability')
    # plt.title('Probability of Each Goal Location')
    # plt.legend()
    # plt.show()


    # plot communication probabilities
    # plt.figure()
    # timesteps = range(1, len(communication)+1)
    # plt.plot(timesteps, communication,marker='',linewidth=2, linestyle=':',label='Probability of Communication')
    # plt.ylim([0, 1])
    # plt.xlabel('Timestep')
    # plt.ylabel('Probability')
    # plt.title('Probability of Communication')
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    four_goals_example()