class Plan:
    def __init__(self, initial_location):
        self._location_at_each_time = {0:initial_location}
        self._communication_at_each_time = [False]
        self._knowledge_at_each_time = {}
        self._rewards_at_each_time = {}

    def set_location(self, t, loc):
        self._location_at_each_time[t] = loc

    def get_location_at_time(self, t):
        return self._location_at_each_time[t]

    def set_knowledge(self, t, knowledge):
        self._knowledge_at_each_time[t] = knowledge

    def get_knowledge_at_time(self, t):
        return self._knowledge_at_each_time[t]

    def get_communication_at_time(self, time):
        return self._communication_at_each_time[time]
        
    def get_duration(self):
        return len(self._location_at_each_time)
