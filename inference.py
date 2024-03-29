from planning import next_optimal_step, next_step_given_beliefs_rewards
from agent import Agent

def infer_communication(agents, environment, t):
    if t < 2:
        return {} # cannot infer communication before actions are taken
    communication_posterior = {}

    total_knowledge = {}
    for agent in agents:
        total_knowledge.update(agent.plan.get_knowledge_at_time(t-1))

    for agent in agents:

        # if an agent's goal changed and visible goals did not change, probability of communication is high.
        prev2_knowledge = agent.plan.get_knowledge_at_time(t-2)
        prev2_loc = agent.plan.get_location_at_time(t-2)
        prev_loc = agent.plan.get_location_at_time(t-1)
        prev_beliefs_given_no_comm = {**agent.get_visible_goals(loc=prev_loc), **prev2_knowledge}
        prev_beliefs_given_comm = {**prev_beliefs_given_no_comm, **total_knowledge}

        cur_loc_given_no_comm, dist = next_step_given_beliefs_rewards(prev_loc, prev_beliefs_given_no_comm, agent.rewards, environment)
        cur_loc_given_comm, dist = next_step_given_beliefs_rewards(prev_loc, prev_beliefs_given_comm, agent.rewards, environment)
        cur_loc = agent.plan.get_location_at_time(t)

        #print(prev_beliefs_given_comm, prev_beliefs_given_no_comm, agent.plan._knowledge_at_each_time, t)

        if cur_loc_given_comm == cur_loc:
            if cur_loc_given_comm == cur_loc_given_no_comm: # communication would not change agent's behavior
                communication_posterior[agent.name] = 0.3
            else: # strong evidence for communication over no-comm
                communication_posterior[agent.name] = 0.9
        else: # strong evidence for no-comm
            communication_posterior[agent.name] = 0.15


    # TODO a better extension: if the agent's goal changed in a way that 
    # could be caused by the other agent's knowledge, then probability of 
    # communication is high
    # give agent some probability of randomly changing its goal or taking a 
    # random step regardless of communication
    # then, in a setting with many goals the probability of communication 
    # can depend on the action/new inferred goal.
    return communication_posterior


def infer_goal(agent, t):
    if t < 1: 
        return {} # cannot infer goal before actions are taken
    # get previous and current location
    prev_loc = agent.plan.get_location_at_time(t-1)
    cur_loc = agent.plan.get_location_at_time(t)

    # inverse planning: get actions agent would have taken on the last timestep given different goals
    rational_goals = []
    for goal_loc in agent.environment.goal_locations:
        next_loc, dist = next_optimal_step(prev_loc, goal_loc, agent.environment.obstacle_map)
        if next_loc == cur_loc:
            rational_goals += [goal_loc]

    # determine goal probabilities
    goal_posterior = {}
    n_rational_goals = len(rational_goals)
    n_goals = len(agent.environment.goal_locations)
    for goal_loc in agent.environment.goal_locations:
        if goal_loc in rational_goals:
            goal_posterior[goal_loc] = 0.9 / n_rational_goals + 0.1 / n_goals
        else:
            goal_posterior[goal_loc] = 0.1 / n_goals

    return goal_posterior