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
    plan = create_plan(environment=env, agents=[agent1, agent2], timesteps=80, cprob=0.0, ctime=6)

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
           #fig = plt.figure()
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

    # plot human inferred goals
    # timesteps = [0,4,5,6,7,11,12,17,22]
    # plt.plot(timesteps, [.63,.63,.6,.1,.03,.03,.03,0,0],marker='',linewidth=2, linestyle=':',label='A')
    # plt.plot(timesteps, [0,0,0,0,0,0,0,.02,0],marker='',linewidth=2, linestyle='-',label='B')
    # plt.plot(timesteps, [.3,.3,.37,.57,.6,.73,.83,.97,1],marker='',linewidth=2, linestyle='--',label='C')
    # plt.plot(timesteps, [.37,.37,.33,.33,.23,.23,.13,.02,0],marker='',linewidth=2, linestyle='-.',label='D')

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
    
    # plot communication probabilities
    plt.figure()
    comm_so_far = [max(communication[:i+1]) for i in range(len(communication))]
    timesteps = range(1, len(communication)+1)
    plt.plot(timesteps, comm_so_far,marker='',linewidth=2, linestyle=':')
    plt.ylim([0, 1])
    #plt.plot(timesteps, 1-communication,marker='',linewidth=2, linestyle='-',label='B')
    plt.xlabel('Timestep')
    plt.ylabel('Probability')
    plt.title('Probability of Communication')


    # plot human inferred communication probabilities
    plt.figure()
    timesteps = [0,4,5,6,7,11,12,17,22]
    #timesteps = [0,5,8,10,14,15,17,21,22,25,30,33]
    #responses_scenario_2 = [[0.1, 0.1, 0.3, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5],
    #                        [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
    #                        [0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    #                        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0, 0, 0, 0],
    #                        [0.3, 0.1, 0.2, 0.2, 0.2, 0.05, 0.05, 0.15, 0.15, 0.15, 0.15, 0.15]]
    responses_scenario_3 = [[0.15, 0.2, 0.2, 0.4, 0.45, 0.8, 0.8, 0.8, 0.9],
                            [0.15, 0.15, 0.15, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
                            [0.15, 0.15, 0.15, 0.5, 0.6, 0.8, 0.9, 0.9, 0.95],
                            [0.0, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
                            [0.15, 0.15, 0.15, 0.8, 0.8, 0.8, 0.9, 0.95, 1.0]]
    plt.plot(timesteps, np.mean(responses_scenario_3, axis=0),marker='',linewidth=2, linestyle=':')
    plt.ylim([0, 1])
    plt.xlabel('Timestep')
    plt.ylabel('Probability')
    plt.title('Probability of Communication')
    plt.show()


if __name__ == "__main__":
    four_goals_example()