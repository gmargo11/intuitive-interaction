from planning import next_optimal_step

def infer_communication(environment, agents, plan):
    history = []
    goal = []
    for t in range(plan._duration):
        location = plan.get_location_at_time(t)
        loc_history += [location]
        goal = infer_goal(loc_history)
        goal_history += [goal]


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

    predictions = {}
    n_rational_goals = len(rational_goals)
    n_goals = len(agent.rewards.keys())
    for goal in agent.rewards.keys():
        if goal in rational_goals:
            predictions[goal] = 0.9 / n_rational_goals + 0.1 / n_goals
        else:
            predictions[goal] = 0.1 / n_goals

    return predictions