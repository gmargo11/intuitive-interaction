class Plan:
    def __init__(self, initial_location):
        self._duration = 1
        self._location_at_each_time = [initial_location]
        self._communication_at_each_time = []
        self._beliefs_at_each_time = []
        self._rewards_at_each_time = []

    def set_next_location(self, loc):
        self._location_at_each_time.append(loc)
        self._duration += 1

    def get_location_at_time(self, time):
        return self._location_at_each_time[time]

    def get_communication_at_time(self, time):
        return self._communication_at_each_time[time]
        
    def get_duration(self):
        return self._duration
