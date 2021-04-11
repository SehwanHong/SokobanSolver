import sys
import collections
import numpy as np
import heapq
import time
import os

class PriorityQueue:
    """
    PriorityQueue is defined using heapq.
    Checks entry which contains priority value, counter, and item.
    Pop the first entry that have the lowest priority value.
    """
    def  __init__(self):
        self.Heap = []
        self.Counter = 0

    def push(self, value, priorityValue):
        entry = (priorityValue, self.Counter, value)
        heapq.heappush(self.Heap, entry)
        self.Counter += 1

    def pop(self):
        (_, _, Value) = heapq.heappop(self.Heap)
        return Value

    def isEmpty(self):
        return len(self.Heap) == 0

"""
Defines the logistic of Sokoban game.
There are some rules that must be implimented.
"""

def stringToGameState(string):
    """
    Gets string input and transfer them into three different values: position of wall, position of boxes, and position of player
    """

    inputs = string[0].split()
    sizeOfSokoban = (int(inputs[0]), int(inputs[1]))

    # read walls from string input
    wallPosition = string[1].split()
    wallPosition.pop(0)
    wallSet = set()
    for i in range(0, len(wallPosition), 2):
        wallSet.add((int(wallPosition[i]), int(wallPosition[i+1])))
        
    # read boxes from string input
    boxPosition = string[2].split()
    boxPosition.pop(0)
    boxSet = set()
    for i in range(0, len(boxPosition), 2):
        boxSet.add((int(boxPosition[i]), int(boxPosition[i+1])))

    # read storages
    goalPosition = string[3].split()
    goalPosition.pop(0)
    goalSet = set()
    for i in range(0, len(goalPosition), 2):
        goalSet.add((int(goalPosition[i]), int(goalPosition[i+1])))

    # read start position
    playerPosition = string[4].split()
    playerPos = (int(playerPosition[0]), int(playerPosition[1]))

    return sizeOfSokoban, wallSet, boxSet, goalSet, playerPos

def PlayerPos(playerPosition):
    """Agent Position is returned"""
    return tuple(playerPosition) # e.g. (2, 2)

def BoxPos(boxPosition):
    """Box position is returned"""
    return tuple(boxPosition) # e.g. ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))

def WallPos(wallPosition):
    """Wall Position is returned"""
    return tuple(wallPosition) # see BoxPos result

def GoalPos(goalPosition):
    """Goal Position is returned"""
    return tuple(goalPosition) # see BoxPos result

def isEndState(boxPosition, goalPosition):
    """Check if boxes are on the goals"""
    return sorted(boxPosition) == sorted(goalPosition)

def isLegalAction(action, playerPosition, boxPosition, wallPosition):
    """Finds out if the action of Player is Legal"""
    xPlayer, yPlayer = playerPosition
    if action[-1].isupper(): # the move was a push
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
    return (x1, y1) not in boxPosition + wallPosition

def legalActions(playerPosition, boxPosition, wallPosition):
    """Get all posible action of players checking if there is the box near by or the wall near by"""
    '''
    action의 구조를 [y이동방향,x이동방향,안 밀었을 때,밀었을 때] 형식에서
    [y이동방향,x이동방향,명령문자열,밀었는가?]로 바꾸었습니다.
    '''
    allActions = [[-1,0,'u','U'],[1,0,'d','D'],[0,-1,'l','L'],[0,1,'r','R']]
    xPlayer, yPlayer = playerPosition
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        if (x1, y1) in boxPosition: # the move was a push
            action.pop(2) # drop the little letter
        else:
            action.pop(3) # drop the upper letter
        if isLegalAction(action, playerPosition, boxPosition, wallPosition):
            legalActions.append(action)
        else: 
            continue     
    return tuple(tuple(x) for x in legalActions) # e.g. ((0, -1, 'l'), (0, 1, 'R'))

def updateState(playerPos, boxPosition, action):
    """Return updated game state after an action is taken"""
    xPlayer, yPlayer = playerPos # the previous position of player
    newPlayerPosition = [xPlayer + action[0], yPlayer + action[1]] # the current position of player
    boxPosition = [list(x) for x in boxPosition]
    if action[-1].isupper(): # if pushing, update the position of box
        boxPosition.remove(newPlayerPosition)
        boxPosition.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
    boxPosition = tuple(tuple(x) for x in boxPosition)
    newPlayerPosition = tuple(newPlayerPosition)
    return newPlayerPosition, boxPosition

def isFailed(boxPosition, wallPosition, goalPosition):
    """This function used to observe if the state is potentially failed, then prune the search"""
    rotatePattern = [[0,1,2,3,4,5,6,7,8],
                    [2,5,8,1,4,7,0,3,6],
                    [0,1,2,3,4,5,6,7,8][::-1],
                    [2,5,8,1,4,7,0,3,6][::-1]]
    flipPattern = [[2,1,0,5,4,3,8,7,6],
                    [0,3,6,1,4,7,2,5,8],
                    [2,1,0,5,4,3,8,7,6][::-1],
                    [0,3,6,1,4,7,2,5,8][::-1]]
    allPattern = rotatePattern + flipPattern

    for boxPos in boxPosition:
        if boxPos not in goalPosition:
            board = [(boxPos[0] - 1, boxPos[1] - 1), (boxPos[0] - 1, boxPos[1]), (boxPos[0] - 1, boxPos[1] + 1), 
                     (boxPos[0], boxPos[1] - 1),     (boxPos[0], boxPos[1])    , (boxPos[0], boxPos[1] + 1), 
                     (boxPos[0] + 1, boxPos[1] - 1), (boxPos[0] + 1, boxPos[1]), (boxPos[0] + 1, boxPos[1] + 1)]
            for pattern in allPattern:
                transformedPosition = [board[i] for i in pattern]
                if transformedPosition[1] in wallPosition and transformedPosition[5] in wallPosition: return True
                elif transformedPosition[1] in boxPosition and transformedPosition[2] in wallPosition and transformedPosition[5] in wallPosition: return True
                elif transformedPosition[1] in boxPosition and transformedPosition[2] in wallPosition and transformedPosition[5] in boxPosition: return True
                elif transformedPosition[1] in boxPosition and transformedPosition[2] in boxPosition and transformedPosition[5] in boxPosition: return True
                elif transformedPosition[1] in boxPosition and transformedPosition[6] in boxPosition and transformedPosition[2] in wallPosition and transformedPosition[3] in wallPosition and newBoard[8] in wallPosition: return True
    return False

def heuristic(playerPosition, boxPosition, goalPosition):
    """A heuristic function to calculate the overall distance between the else boxes and the else goals"""
    distance = 0
    completes = set(goalPosition) & set(boxPosition)
    sortedBoxPosition = list(set(boxPosition).difference(completes))
    sortedGoalPosition = list(set(goalPosition).difference(completes))
    for i in range(len(sortedBoxPosition)):
        distance += (abs(sortedBoxPosition[i][0] - sortedGoalPosition[i][0])) + (abs(sortedBoxPosition[i][1] - sortedGoalPosition[i][1]))
    return distance

def cost(actions):
    """A cost function"""
    return len([x for x in actions if x.islower()])

def aStarSearch(boxSet, playerPos, wallPos, goalPos):
    """Implement aStarSearch approach"""
    beginBox = BoxPos(boxSet)
    beginPlayer = PlayerPos(playerPos)

    start_state = (beginPlayer, beginBox)
    frontier = PriorityQueue()
    frontier.push([start_state], heuristic(beginPlayer, beginBox, goalPos))
    exploredSet = set()
    actions = PriorityQueue()
    '''
    기존 코드는 actions에서 [0] + [이동] 형태로 저장되던 것을
    사용하지 않는 0번 인덱스를 cost, 1,2,3,4번 인덱스의 값을 각각 UDLR의 카운트로 사용
    '''
    actions.push([" "], heuristic(beginPlayer, start_state[1], goalPos))
    while (not frontier.isEmpty()):
        node = frontier.pop()
        node_action = actions.pop()
        if isEndState(node[-1][-1], goalPos):
            sequenceOfAction = listNodeActionToString(node_action[1:])
            print(sequenceOfAction);
            break
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            """ cost 계산을 원방식으로 돌렸습니다 """
            Cost = cost(node_action[1:])
            """ legalAction 은 변경된 방식을 유지했습니다. """
            for action in legalActions(node[-1][0], node[-1][1], wallPos):
                newPlayerPosition, newBoxPosition = updateState(node[-1][0], node[-1][1], action)
                if isFailed(newBoxPosition, wallPos, goalPos):
                    continue
                Heuristic = heuristic(newPlayerPosition, newBoxPosition, goalPos)
                frontier.push(node + [(newPlayerPosition, newBoxPosition)], Heuristic + Cost) 
                actions.push(node_action + [action[-1]], Heuristic + Cost)

def listNodeActionToString(listOfAction):
    NodeAction = ','.join(listOfAction).replace(',','')
    prev = NodeAction[0].upper()
    count = 0
    nodeActionString = ''
    for x in NodeAction.upper():
        if (prev == x):
            count += 1
        else:
            nodeActionString += "{} {} ".format(count, prev);
            prev = x;
            count = 1;
    nodeActionString += "{} {}".format(count, prev);
    return nodeActionString

"""Read command"""
def readCommand(argv):
    if (type(argv) != str):
        raise SyntaxError("Received that are not string.")
    with open('./'+ argv,"r") as f:
        fileString = f.readlines()
    return fileString

""" 모든 코드를 불러오는 방식을 변경시켰습니다. """

if __name__ == '__main__':
    sokobanString = readCommand(sys.argv[1])
    #sokobanString = readCommand('sokoban00.txt')
    sizeOfSokoban, wallSet, boxSet, goalSet, playerPos = stringToGameState(sokobanString)
    wallPos = WallPos(wallSet)
    goalPos = GoalPos(goalSet)
    aStarSearch(boxSet, playerPos, wallPos, goalPos)
