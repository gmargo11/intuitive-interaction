from environment import Environment
from agent import Agent
from plan import Plan
from planning import create_plan
from inference import infer_communication, infer_goal
from utils import display_map_state, display_belief_state, display_inferred_goals

import numpy as np
import matplotlib.pyplot as plt



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
    # if endpoint is (inf,inf), obstacle reaches boundary.
    # endpoints are 0 spaces that are diagonal to end of obstacles
    obstacle_endpoints = [((7,2), (float('inf'), float('inf')))]
    env = Environment(obstacle_map=obstacle_map, obstacle_endpoints=obstacle_endpoints, goal_locations=goal_locations)


    # define agents
    agent1 = Agent(rewards={goal_locations[0]: 3, goal_locations[1]: 10}, initial_location=(3, 7), initial_beliefs=np.array([[0.5, 0.5], [0.5, 0.5]]), environment=env)
    agent2 = Agent(rewards={goal_locations[0]: 5, goal_locations[1]: 8}, initial_location=(11, 7), initial_beliefs=np.array([[0.5, 0.5], [0.5, 0.5]]), environment=env)

    #agent1.plan.set_next_location((4, 7))
    #agent2.plan.set_next_location((10, 7))

    #for t in range(agent1.plan.get_duration()):
    #    display_map_state(environment=env, agents=[agent1, agent2], t=t)
    #    display_belief_state(agent=agent1)

    #plt.show()

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
                            [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    goal_locations = [(3, 12), (7, 12), (12, 12)]
    # if endpoint is (inf,inf), obstacle reaches boundary.
    # endpoints are 0 spaces that are diagonal to end of obstacles
    env = Environment(obstacle_map=obstacle_map, goal_locations=goal_locations)


    # define agents
    agent1 = Agent(rewards={goal_locations[0]: 3, goal_locations[1]: 5, goal_locations[2]: 10}, initial_location=(7, 7), initial_beliefs=np.array([[0.5, 0.5], [0.5, 0.5]]), environment=env)
    agent2 = Agent(rewards={goal_locations[0]: 5, goal_locations[1]: 8, goal_locations[2]: 7}, initial_location=(11, 7), initial_beliefs=np.array([[0.5, 0.5], [0.5, 0.5]]), environment=env)

    #agent1.plan.set_next_location((4, 7))
    #agent2.plan.set_next_location((10, 7))

    #for t in range(agent1.plan.get_duration()):
    #    display_map_state(environment=env, agents=[agent1, agent2], t=t)
    #    display_belief_state(agent=agent1)

    #plt.show()

    # generate plans
    plan = create_plan(environment=env, agents=[agent1, agent2], timesteps=30, cprob=0.2)
    for p in plan:
        print(plan[p]._location_at_each_time)

    plt.figure()
    for t in range(agent1.plan.get_duration()):
         plt.cla()
         plt.subplot(1, 2, 1)
         display_map_state(environment=env, agents=[agent1, agent2], t=t)
         plt.subplot(1, 2, 2)
         display_inferred_goals(infer_goal(agent1, t))
         plt.waitforbuttonpress()
    plt.show()

    # infer communication
    #communication_prob_at_each_time = infer_communication(environment=env, agents=[agent1, agent2], plan=plan)



if __name__ == "__main__":
    three_goals_example()