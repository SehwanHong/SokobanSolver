import sys
import collections
import numpy as np
import heapq
import time

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
        wallSet.add((int(wallPosition[i+1]), int(wallPosition[i])))
        
    # read boxes from string input
    boxPosition = string[2].split()
    boxPosition.pop(0)
    boxSet = set()
    for i in range(0, len(boxPosition), 2):
        boxSet.add((int(boxPosition[i+1]), int(boxPosition[i])))

    # read storages
    goalPosition = string[3].split()
    goalPosition.pop(0)
    goalSet = set()
    for i in range(0, len(goalPosition), 2):
        goalSet.add((int(goalPosition[i+1]), int(goalPosition[i])))

    # read start position
    playerPosition = string[4].split()
    playerPos = (int(playerPosition[1]), int(playerPosition[0]))

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
    if action[-1] == 1: # the move was a push
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
    return (x1, y1) not in boxPosition or (x1,y1) not in wallPosition

def legalActions(playerPosition, boxPosition, wallPosition):
    """Get all posible action of players checking if there is the box near by or the wall near by"""
    '''
    action의 구조를 [y이동방향,x이동방향,안 밀었을 때,밀었을 때] 형식에서
    [y이동방향,x이동방향,명령문자열,밀었는가?]로 바꾸었습니다.
    '''
    allActions = [[-1,0,'U',0],[1,0,'D',0],[0,-1,'L',0],[0,1,'R',0]]
    xPlayer, yPlayer = playerPosition
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        '''
        밀었을 때에 안 밀린 문자열을 리스트에서 삭제하고, 안 밀렸을 땐 밀린 문자열을 삭제하는 루틴에서
        밀렸을 때 1로 바꾸는 것으로 변경했습니다.
        '''
        if (x1, y1) in boxPosition: # the move was a push
            action[3] = 1
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
    if action[-1] == 1: # if pushing, update the position of box
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
    actions.push([0,0,0,0,0], heuristic(beginPlayer, start_state[1], goalPos))
    while frontier:
        node = frontier.pop()
        node_action = actions.pop()
        if isEndState(node[-1][-1], goalPos):
            print(' '.join([f'{cnt} {action}' for cnt,action in zip(node_action[1:],'UDLR')]))
            break
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            '''
            기존 cost를 연산하는 코드에서 불러오는 것으로 변경
            '''
            Cost = node_action[0]
            '''
            legalActions함수가 변경되었습니다.
            자세한 사항은 legalActions함수 주석을 확인해주세요
            '''
            for action in legalActions(node[-1][0], node[-1][1], wallPos):
                newPlayerPosition, newBoxPosition = updateState(node[-1][0], node[-1][1], action)
                if isFailed(newBoxPosition, wallPos, goalPos):
                    continue
                Heuristic = heuristic(newPlayerPosition, newBoxPosition, goalPos)
                frontier.push(node + [(newPlayerPosition, newBoxPosition)], Heuristic + Cost)

                '''
                cost를 저장하기 위해 새롭게 계산
                '''
                new_cost = Cost
                if action[-1] == 0:
                    new_cost += 1

                '''
                기존 명령에 현재 명령을 추가하여 카운트를 갱신합니다
                '''
                new_action = node_action[1:]
                new_action['UDLR'.index(action[-2])] += 1
                

                actions.push([new_cost] + new_action, Heuristic + Cost)

"""Read command"""
def readCommand(argv):
    if (len(argv) >= 2):
        raise SyntaxError("received more than one arguement, Only accept one argument of txt file")
    with open('./'+ argv[0],"r") as f:
        fileString = f.readlines()
    return fileString


if __name__ == '__main__':
    #sokobanString = readCommand(sys.argv[1:])
    sokobanString = readCommand(['sokoban01.txt'])
    sizeOfSokoban, wallSet, boxSet, goalSet, playerPos = stringToGameState(sokobanString)
    wallPos = WallPos(wallSet)
    goalPos = GoalPos(goalSet)
    aStarSearch(boxSet, playerPos, wallPos, goalPos)
