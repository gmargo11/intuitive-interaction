def infer_communication(environment, agents, plan):
    history = []
    goal = []
    for t in range(plan._duration):
        location = plan.get_location_at_time(t)
        loc_history += [location]
        goal = infer_goal(loc_history)
        goal_history += [goal]
        

def infer_goal(loc_history):
    # compate