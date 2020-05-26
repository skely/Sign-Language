import math
def comp(trajectory,start,end):
    dists = []
    for i in range(len(trajectory[1])):
        sX = trajectory[start][i][0]
        sY = trajectory[start][i][1]
        sZ = trajectory[start][i][2]
        eX = trajectory[end][i][0]
        eY = trajectory[end][i][1]
        eZ = trajectory[end][i][2]
        dist = math.sqrt(((sX-eX)**2)+((sY-eY)**2)+((sZ-eZ)**2))
        dists.append(dist)
    
    return dists
