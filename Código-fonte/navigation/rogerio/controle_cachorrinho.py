
class PositioningControl():

    def __init__(self, center,centerThreshold, distance, distanceThreshold):
        self.center = center
        self.distance = distance
        self.centerLimits = [center - centerThreshold, center + centerThreshold]
        self.distanceLimits = [distance - distanceThreshold, distance + distanceThreshold]

    def offCenter(self, position):
        if(self.centerLimits[0] <= position <= self.centerLimits[1]):
            return False
        else:
            return True

    def offDistance(self, distance):
        if(self.distanceLimits[0] <= distance <= self.distanceLimits[1]):
            return False
        else:
            return True
