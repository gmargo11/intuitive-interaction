from planning import next_optimal_step

def infer_communication(agents, t):
    if t < 2:
        return {} # cannot infer communication before actions are taken
    communication_posterior = {}
    for agent in agents:
        # infer agent goals at the previous time and this time
        prev_goal = infer_goal(agent, t-1)
        cur_goal = infer_goal(agent, t)

        # if an agent's goal changed and visible goals did not change, probability of communication is high.
        prev_visible = agent.get_visible_goals(loc=agent.plan.get_location_at_time(t-2))
        cur_visible = agent.get_visible_goals(loc=agent.plan.get_location_at_time(t-1))

        if prev_goal != cur_goal and prev_visible == cur_visible:
            print(prev_visible, cur_visible)
            communication_posterior[agent.name] = 0.9
        else:
            communication_posterior[agent.name] = 0.1


    # TODO a better extension: if the agent's goal changed in a way that could be caused by the other agent's knowledge, then probability of communication is high
    # give agent some probability of randomly changing its goal or taking a random step regardless of communication
    # then, in a setting with many goals the probability of communication can depend on the action/new inferred goal.
    return communication_posterior


def infer_goal(agent, t):
    if t < 1: 
        return {} # cannot infer goal before actions are taken
    # get previous and current location
    prev_loc = agent.plan.get_location_at_time(t-1)
    cur_loc = agent.plan.get_location_at_time(t)

    # inverse planning: get actions agent would have taken on the last timestep given different goals
    rational_goals = []
    for goal in agent.rewards.keys():
        next_loc, dist = next_optimal_step(prev_loc, goal, agent.environment.obstacle_map)
        print(next_loc, cur_loc)
        if next_loc == cur_loc:
            rational_goals += [goal]

    goal_posterior = {}
    n_rational_goals = len(rational_goals)
    n_goals = len(agent.rewards.keys())
    for goal in agent.rewards.keys():
        if goal in rational_goals:
            goal_posterior[goal] = 0.9 / n_rational_goals + 0.1 / n_goals
        else:
            goal_posterior[goal] = 0.1 / n_goals

    return goal_posterior