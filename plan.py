class Plan:
    def __init__(self):
        self._duration = 0
        self._location_at_each_time = []
        self._communication_at_each_time = []
        self._beliefs_at_each_time = []
        self._rewards_at_each_time = []

    def get_location_at_time(self, time):
        return self._location_at_each_time[time]

    def get_communication_at_time(self, time):
        return self._communication_at_each_time[time]
        

