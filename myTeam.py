
# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import sys
sys.path.append('teams/aimirant/')

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
import capture
from game import Actions

####PIECE OF CODE TO COUNT HOW LONG AN ALGORITHM TAKES TO RUN
start = time.time()
#insert code

#print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

import numpy as np
#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'offensiveAgent', second = 'DefensiveAgent', numTraining=0):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.
  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class AIagent(CaptureAgent):
  def __init__(self, index):
    CaptureAgent.__init__(self, index)
    self.alpha = 0.2
    self.reward = 1
    self.gamma = 0.8
    self.epsilon = 0.01
    self.qSa = None
    self.Q_present = 0
    self.numTrainig = 10
 

    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

  def registerInitialState(self, gameState):
    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

    #Stablishes the boundary of my territory
    if self.red:
      self.edge = (gameState.data.layout.width - 2) // 2
    else:
      self.edge = ((gameState.data.layout.width - 2) // 2) + 1
    
    self.boundary = []
    for i in range(1, gameState.data.layout.height - 1):
      if not gameState.hasWall(self.edge, i):
        self.boundary.append((self.edge, i))

    

    '''
    Your initialization code goes here, if you need any.
    '''

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action) #Successor basically becomes a STATE and it contains all the info a gameState might have
    pos = successor.getAgentState(self.index).getPosition() #position for THIS agent
    if pos != nearestPoint(pos): #this is because apparently the GHOSTS move much faster than normal pacmans, so it creates a dissonance
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    ##are these the feauters we should use in our Q learning approximation? if so we want enemy location, walls, food, capsules
    features = util.Counter() #counter is a utility to create dictionaries with many functionalities applicable to Q learning
    successor = self.getSuccessor(gameState, action) # successor is the GAMESTATE that happens after action
    features['successorScore'] = self.getScore(successor) #defining a member of the dictionary and assigning the score
    features['walls'] = gameState.hasWalls()

    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}


  def aStar(self, gameState, heuristic, goalState, timerStart):
      """Search the node that has the lowest combined cost and heuristic first."""
      self.start = gameState.getAgentPosition(self.index)
      self.food = self.getFood(gameState)
      myPQ = util.PriorityQueue()
      startState = (gameState, self.food)
      prevGameState = startState
      startNode = (startState, '', 0, [])
      myPQ.push(startNode, heuristic(gameState, prevGameState))
      visited = set()
      best_g = dict()
      while not myPQ.isEmpty():
        currGameState, action, cost, path  = myPQ.pop() #where currGameState[0] is the gameState and currGameState[1] is the food.
        if (not currGameState in visited) or cost < best_g.get(currGameState):
          visited.add(currGameState)
          best_g[currGameState] = cost
          #self.f.write("time inside astar search:%.4f" % (time.time() - self.timerStart))
          # check if the current state is a goal state or the evaluation time is getting critically high (threshold 0.7s).
          if goalState(currGameState, prevGameState) or (time.time() - timerStart >= 0.85):
            path = path + [(currGameState, action)]
            actions = [action[1] for action in path]
            del actions[0]
            return actions

          for currAction in currGameState[0].getLegalActions(self.index):
            if (time.time() - timerStart < 0.9):
              successor = self.getSuccessor(currGameState[0], currAction)
              if self.checkIfLocationHasOpponent(successor.getAgentPosition(self.index),currGameState[0]):
                newNode = ((successor, self.getFood(successor)), currAction, cost + 1, path + [(currGameState, action)] )
                myPQ.push(newNode, heuristic(successor, currGameState) + cost + 1)
          prevGameState = currGameState   
      return ['Stop']
      #util.raiseNotDefined()

  def checkIfLocationHasOpponent(self, succPos, gameState):
    opponent = self.getOpponents(gameState)
    if (not gameState.getAgentState(opponent[0]).isPacman and gameState.getAgentPosition(opponent[0]) != None
          and gameState.getAgentState(opponent[0]).scaredTimer == 0):
          if succPos in gameState.getAgentPosition(opponent[0]):
            print("avoid ghost")
            return False
          else:
            return True
    elif (not gameState.getAgentState(opponent[1]).isPacman and gameState.getAgentPosition(opponent[1]) != None
          and gameState.getAgentState(opponent[1]).scaredTimer == 0):
          if succPos in gameState.getAgentPosition(opponent[1]):
            return False
          else:
            return True
    else:
      return True

class offensiveAgent(AIagent):
 
  def __init__(self,index):
    AIagent.__init__(self, index)
    self.actions = []
    #Creates the boundary
    self.edge=None
    self.boundary=[]
    #self.f = open("logOffense.txt", "w")
    self.timerStart = time.time()
    self.timerCountdown = time.time()
    self.goingHome = False
    self.onPath = False
    
    


  def registerInitialState(self, gameState):
    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    #Establishes the boundary of my territory
    if self.red:
      self.edge = (gameState.data.layout.width - 2) // 2
    else:
      self.edge = ((gameState.data.layout.width - 2) // 2) + 1
    
    self.boundary = []
    for i in range(1, gameState.data.layout.height - 1):
      if not gameState.hasWall(self.edge, i):
        self.boundary.append((self.edge, i))

  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}


  # method checks if the agent is inside his own territory.
  def inMyTerritory(self, gameState):
    myPos=gameState.getAgentPosition(self.index)
    if self.red and myPos[0]<=self.edge:
      return True
    elif self.red and myPos[0]>self.edge:
      return False
    elif not self.red and myPos[0]>=self.edge:
      return True
    elif not self.red and myPos[0]<self.edge:
      return False
  
  def inMyTerritoryIndex(self, myPos):
    if self.red and myPos[0]<=self.edge:
      return True
    elif self.red and myPos[0]>self.edge:
      return False
    elif not self.red and myPos[0]>=self.edge:
      return True
    elif not self.red and myPos[0]<self.edge:
      return False


  def isGoalStateEscape(self,currGameState, prevGameState):
    myPos = currGameState[0].getAgentPosition(self.index)

    return myPos in self.boundary

  def escapeHeuristic(self, successorState, currGameState):
    x,y = currGameState[0].getAgentPosition(self.index)
    bestDist=self.getMazeDistance(currGameState[0].getAgentPosition(self.index),self.boundary[0])
    
    for boundaryPoint in self.boundary:
      dist= self.getMazeDistance((x,y),boundaryPoint)
      if dist<bestDist:
        bestDist=dist
    return bestDist
    

  def isGoalState(self, currGameState, prevGameState):
    # if this kicks in the agent won the game. However this goalstate should never kick in,
    # as it is already taken care of in chooseAction(). Still here just in case.
    if(currGameState[1].count() == 2):
      return True

    x,y = currGameState[0].getAgentPosition(self.index)
    foodPositions = self.getFood(prevGameState[0]).asList()
    capsulePositions = self.getCapsules(prevGameState[0])
    foodPositions = foodPositions + capsulePositions
    # The goal is if pacman's current position is a location where there is
    # a piece of food or a capsule. 
    return (x, y) in foodPositions


  def searchHeuristic(self, successorState, currGameState):
    #capsules = self.getCapsules(currGameState[0])
    foodGrid = self.getFood(currGameState[0]).asList()
    #foodGrid = foodGrid + capsules
    #foodGrid = foodGrid + capsules
    h = 0
    # to find the optimal paths to the food.
    if not len(foodGrid) == 0:
      maxDis = 9999
      for food in foodGrid:
        maxDis = min(self.getMazeDistance(successorState.getAgentPosition(self.index), food), maxDis)
      h = maxDis
    # in order to consider the opponent ghosts to avoid them on the path towards food.
    opponents = self.getOpponents(successorState)
    distances = successorState.getAgentDistances()
    minPacDis = 9999
    for opponent in opponents:
      if(not successorState.getAgentState(opponent).isPacman):
        if(successorState.getAgentPosition(opponent) != None):
          if(successorState.getAgentState(opponent).scaredTimer > 0):
            minPacDis = 0  
          minPacDis = min(self.getMazeDistance(successorState.getAgentPosition(self.index), successorState.getAgentPosition(opponent)), minPacDis)
        else:
          minPacDis = min(distances[opponent], minPacDis)

    # combine avoidance of opponent ghosts and search for food into heuristic value.
    if(minPacDis != 0):
      h = 0.6 * h + 0.4 * 100 * (1/minPacDis)

    return h

  def findCapsuleHeuristic(self, successorState, currGameState):
    #x,y = currGameState[0].getAgentPosition(self.index)
    capsules = self.getCapsules(currGameState[0])
    h = 0
    if not len(capsules) == 0:
      maxDis = 0
      for capsule in capsules:
        maxDis = max(self.getMazeDistance(successorState.getAgentPosition(self.index), capsule), maxDis)
      h = maxDis
    return h

  def capsuleGoalState(self, currGameState, prevGameState):
    # if this kicks in the agent won the game. However this goalstate should never kick in,
    # as it is already taken care of in chooseAction(). Still here just in case.
    x,y = currGameState[0].getAgentPosition(self.index)
    capsulePositions = self.getCapsules(prevGameState[0])
    # The goal is if pacman's current position is a location where there is
    # a piece of food or a capsule. 
    return (x, y) in capsulePositions

  ####################################################################################################################
  
  
  
  def chooseAction(self, gameState):
    # sets starting point for timer in order to avoid time-out moves.    
    self.timerStart = time.time()
    if(not gameState.getAgentState(self.index).isPacman and gameState.getAgentState(self.index).getDirection() == 'Stop'):
      self.actions = []


    foodLeft = len(self.getFood(gameState).asList())
    capsulesLeft = len(self.getCapsules(gameState))

    if(self.timerCountdown - time.time() >= 1150 and gameState.getAgentState(self.index).numCarrying > 0 and self.goingHome == False):
      self.actions = self.aStar(gameState, self.escapeHeuristic, self.isGoalStateEscape, self.timerStart)
      self.goingHome = True

    # the game is won, when only 2 foods are left, therefore head into own territory in order to deposit food.
    if foodLeft <= 2:
      actions = gameState.getLegalActions(self.index)
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(gameState.getInitialAgentPosition(self.index),pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction


    else:
      opponent = self.getOpponents(gameState)


      if(not gameState.getAgentState(opponent[0]).isPacman and gameState.getAgentPosition(opponent[0]) != None
          and gameState.getAgentState(opponent[0]).scaredTimer == 0): #and self.onPath == False):
        if(capsulesLeft > 0):
          #self.onPath = True
          self.actions = self.aStar(gameState, self.findCapsuleHeuristic, self.capsuleGoalState, self.timerStart)
        if gameState.getAgentState(self.index).numCarrying > 0:
          if self.inMyTerritory(gameState): 
            self.actions = self.aStar(gameState, self.searchHeuristic, self.isGoalState, self.timerStart)
          else:
            # Generate new plan to escape to home.
            self.actions = self.aStar(gameState, self.escapeHeuristic, self.isGoalStateEscape, self.timerStart)
      elif(not gameState.getAgentState(opponent[1]).isPacman and gameState.getAgentPosition(opponent[1]) != None
          and gameState.getAgentState(opponent[1]).scaredTimer == 0): # and self.onPath == False):
        if(capsulesLeft > 0):
          #self.onPath = True
          self.actions = self.aStar(gameState, self.findCapsuleHeuristic, self.capsuleGoalState, self.timerStart)
        if gameState.getAgentState(self.index).numCarrying > 0:
          if self.inMyTerritory(gameState):
            self.actions = self.aStar(gameState, self.searchHeuristic, self.isGoalState, self.timerStart)
          else:
            # Generate new plan to escape to home.
            self.actions = self.aStar(gameState, self.escapeHeuristic, self.isGoalStateEscape, self.timerStart)
      elif(not gameState.getAgentState(opponent[1]).isPacman and gameState.getAgentPosition(opponent[1]) != None
          and gameState.getAgentState(opponent[1]).scaredTimer == 0 and gameState.getAgentState(self.index).numCarrying == 0):
          self.actions = self.aStar(gameState, self.searchHeuristic, self.isGoalState, self.timerStart)
      elif(not gameState.getAgentState(opponent[0]).isPacman and gameState.getAgentPosition(opponent[0]) != None
          and gameState.getAgentState(opponent[0]).scaredTimer == 0 and gameState.getAgentState(self.index).numCarrying == 0):
          self.actions = self.aStar(gameState, self.searchHeuristic, self.isGoalState, self.timerStart)
      elif(self.actions == []):
        self.actions = self.aStar(gameState, self.searchHeuristic, self.isGoalState, self.timerStart)


      if(self.actions == []):
        return 'Stop'
      action = self.actions[0]
      if(action not in gameState.getLegalActions(self.index)):
        return 'Stop'
      self.actions.pop(0)
      #if self.actions == []:
       # self.onPath = False

      return action


class DefensiveAgent(AIagent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """
  def __init__(self,index):
    AIagent.__init__(self, index)
    self.actions = []
    self.edge=None
    self.boundary=[]
    self.targetedBastard=None
    self.diff = None# Food our opponent has just eaten in order to locate opponent, when not in vision.
    self.timerStart = time.time()
    #self.f = open("logDefense.txt", "w")

    self.weights = util.Counter()
    self.features = util.Counter()
    self.edge = None
    # Value initialisation technique:
    self.weights = {'bias': 0, 'numInvaders': 0, 'onDefense': 3.005230727714532, 'invaderDistance': -100, 'stop': -15, 'reverse': -60.41612004318696, 'foodRatio': 4.814181459755859,'inTerritory':-500}
    self.lastState = None
    self.myInitialFood=0
    self.nordPatrol=None
    self.southPatrol=None

  def registerInitialState(self, gameState):
    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    self.start = gameState.getAgentPosition(self.index)
    self.myInitialFood=self.getFoodYouAreDefending(gameState).count()
    CaptureAgent.registerInitialState(self, gameState)

    #Establishes the boundary of my territory
    if self.red:
      self.edge = (gameState.data.layout.width - 2) // 2
    else:
      self.edge = ((gameState.data.layout.width - 2) // 2) + 1
    
    self.boundary = []
    for i in range(1, gameState.data.layout.height - 1):
      if not gameState.hasWall(self.edge, i):
        self.boundary.append((self.edge, i))
    
    self.actions = self.aStar(gameState, self.defensiveHeuristic, self.isGoalStateBoundary, self.timerStart)
    
    #######Defines nord and south patrol spots################################
    lenghtBound=len(self.boundary) 
    self.nordPatrol=(self.edge, self.boundary[int(lenghtBound *0.75)][1])
    self.southPatrol=(self.edge, self.boundary[int(lenghtBound *0.25)][1])
    #print('Nord boundary is: ', self.nordPatrol)
    #print('South boundary is: ', self.southPatrol)
    ##########################################################################

    self.q = util.Counter()
    self.featExtractor = ExtractorDefensive()
    self.startEpisode()


  def chooseAction(self, gameState):
    self.timerStart = time.time()

    opponent = self.getOpponents(gameState)
    myPos= gameState.getAgentPosition(self.index)
    opponent1=gameState.getAgentPosition(opponent[0])
    opponent2=gameState.getAgentPosition(opponent[1])
    invaders=False
    if(opponent1!=None or opponent2!=None):
      invaders=True

    if opponent1 !=None:
      dist1=self.getMazeDistance(myPos,gameState.getAgentPosition(opponent[0]))
    if opponent2 !=None:
      dist2=self.getMazeDistance(myPos,gameState.getAgentPosition(opponent[1]))
    
    currDefendingFoodList = self.getFoodYouAreDefending(gameState).asList()
    currDefendingFood = len(currDefendingFoodList)
    if self.getPreviousObservation() != None:
      prevDefendingFoodList =self.getFoodYouAreDefending(self.getPreviousObservation()).asList()
      prevDefendingFood = len(prevDefendingFoodList)
    else:
      prevDefendingFood = 0
    
    if(self.actions==[] and not invaders):#######################################################
      position=gameState.getAgentPosition(self.index)
      distNord=self.getMazeDistance(position,self.nordPatrol)
      if distNord==0:
        #print('calling south patrol')
        self.actions=self.aStar(gameState, self.patrolSouthHeuristic, self.isGoalStateSouth, self.timerStart)
      else:
        #print('calling nord patrol')
        self.actions=self.aStar(gameState, self.patrolNordHeuristic, self.isGoalStateNord, self.timerStart)   
    elif(self.actions == [] and  myPos not in self.boundary): 
      #if myPos not in self.boundary or not invaders : #if actions is empty and I'm not in boundary
      self.actions = self.aStar(gameState, self.defensiveHeuristic, self.isGoalStateBoundary, self.timerStart)
    elif(gameState.getAgentState(opponent[0]).isPacman and gameState.getAgentPosition(opponent[0]) != None):
      if self.inMyTerritory(gameState): 
        self.targetedBastard=gameState.getAgentPosition(opponent[0])
        self.actions = self.aStar(gameState, self.killTheBastardHeuristic, self.isGoalStateBastard, self.timerStart)
      else:
        #Generate plan to go to boundary.
        self.actions = self.aStar(gameState, self.defensiveHeuristic, self.isGoalStateBoundary, self.timerStart)      
    elif(gameState.getAgentState(opponent[1]).isPacman and gameState.getAgentPosition(opponent[1]) != None):
      if self.inMyTerritory(gameState):
        self.targetedBastard=gameState.getAgentPosition(opponent[1])
        self.actions = self.aStar(gameState, self.killTheBastardHeuristic, self.isGoalStateBastard, self.timerStart)
      else:
        #Generate plan to go to boundary.
        self.actions = self.aStar(gameState, self.defensiveHeuristic, self.isGoalStateBoundary, self.timerStart)
    elif currDefendingFood < prevDefendingFood:
      self.diff = list(set(prevDefendingFoodList).difference(currDefendingFoodList))
      self.actions = self.aStar(gameState, self.locateEatenFoodHeuristic, self.isGoalStateEatenFood, self.timerStart)
    elif self.actions==[]:       
      # use q learning approach in order to patrol the boundary.
      return self.chooseActionQ(gameState)

    if(self.actions == []):
      return 'Stop'
    action = self.actions[0]
    if(action not in gameState.getLegalActions(self.index)):
      return 'Stop'
    self.actions.pop(0)

    return action
    
  ####################################################################
  # The goal state is being in the boundary
  def isGoalStateNord(self, currGameState, prevGameState):
    myPos = currGameState[0].getAgentPosition(self.index)
    return myPos==self.nordPatrol

  # The goal state is being in the boundary
  def isGoalStateSouth(self, currGameState, prevGameState):
    myPos = currGameState[0].getAgentPosition(self.index)
    return myPos==self.southPatrol
  ####################################################################

  # The goal state is being in the boundary
  def isGoalStateBoundary(self, currGameState, prevGameState):
    myPos = currGameState[0].getAgentPosition(self.index)
    return myPos in self.boundary

  # The goal is being at the same position as the invader's
  def isGoalStateBastard(self, currGameState, prevGameState):
    myPos = currGameState[0].getAgentPosition(self.index)
    if currGameState[0].getAgentState(self.index).scaredTimer!=0:
      distToBastard=util.manhattanDistance(myPos,self.targetedBastard)
      return self.inMyTerritory(currGameState[0]) and distToBastard==2
    else:
      return self.inMyTerritory(currGameState[0]) and myPos == self.targetedBastard

    # The goal is being at the same position as the invader's
  def isGoalStateEatenFood(self, currGameState, prevGameState):
    myPos = currGameState[0].getAgentPosition(self.index)
    return self.inMyTerritory(currGameState[0]) and myPos == self.diff[0]

  def inMyTerritory(self, gameState):
    myPos=gameState.getAgentPosition(self.index)
    if self.red and myPos[0]<=self.edge:
      return True
    elif self.red and myPos[0]>self.edge:
      return False
    elif not self.red and myPos[0]>=self.edge:
      return True
    elif not self.red and myPos[0]<self.edge:
      return False

    
# Tells you how close you are from the nearest position that belongs to the boundary
  def defensiveHeuristic(self, successorState, currGameState):
    x,y = currGameState[0].getAgentPosition(self.index)
    middleBoundary=len(self.boundary)//2
    bestDist=self.getMazeDistance(currGameState[0].getAgentPosition(self.index),self.boundary[middleBoundary])
    return bestDist


  def killTheBastardHeuristic(self, successorState, currGameState):
    myPos = successorState.getAgentPosition(self.index)
    opponent = self.getOpponents(successorState)
    distNearestBastard=util.manhattanDistance(myPos,self.targetedBastard)

    return distNearestBastard 

  def locateEatenFoodHeuristic(self, successorState, currGameState):
    myPos = successorState.getAgentPosition(self.index)
    distFoodEaten = util.manhattanDistance(myPos,self.diff[0])
    return distFoodEaten
  
  ####################################################################################################
  def patrolNordHeuristic(self, successorState, currGameState):
    x,y = currGameState[0].getAgentPosition(self.index)
    goal=self.nordPatrol
    bestDist=self.getMazeDistance(currGameState[0].getAgentPosition(self.index),self.nordPatrol)
    if(not self.inMyTerritory(currGameState[0])):
      bestDist+=5

    return bestDist

  def patrolSouthHeuristic(self, successorState, currGameState):
    x,y = currGameState[0].getAgentPosition(self.index)
    goal=self.nordPatrol
    bestDist=self.getMazeDistance(currGameState[0].getAgentPosition(self.index),self.southPatrol)
    if(not self.inMyTerritory(currGameState[0])):
      bestDist+=5
    return bestDist
  ####################################################################################################

###################################Q-LEARNING FOR DEFENSIVE AGENT###############################################################

  def inMyTerritoryIndex(self, myPos):
    if self.red and myPos[0]<(self.edge-1):
      return True
    elif self.red and myPos[0]>=(self.edge-1):
      return False
    elif not self.red and myPos[0]>(self.edge+1):
      return True
    elif not self.red and myPos[0]<=(self.edge+1):
      return False
  
  def startEpisode(self):
    """
      Called by environment when new episode is starting
    """
    self.lastState = None
    self.lastAction = None
    self.episodeRewards = 0.0
    #print('\nweights of episode are: ',self.weights)

  def getWeights(self):
    return self.weights


  def getQValue(self, gameState, action):

    features = self.featExtractor.getFeatures(gameState, action, self.index,self)
    #(features)
    qvalue = 0

    for feat in features.keys():
      #print(self.weights[feat])
      qvalue = qvalue + self.weights[feat] * features[feat]

    return qvalue

  def computeValueFromQValues(self, gameState):

    legalActions = gameState.getLegalActions(self.index)
    if not legalActions:
      return 0
    return max([self.getQValue(gameState, legalAction) for legalAction in legalActions])



  def computeActionFromQValues(self, gameState):
    count = util.Counter()

    legalActions = gameState.getLegalActions(self.index)
    if not legalActions:
      return None
    else:
      for legalAction in legalActions:
        count[(gameState,legalAction)] = self.getQValue(gameState, legalAction)
      if len(count) == 1:
        return (list(count.keys()))[0][1]
      return count.argMax()[1]


  def getAction0(self, state):
    action = self.getAction1(state)
    self.doAction(state,action)
    return action

  def getAction1(self, gameState):
    legalActions = gameState.getLegalActions(self.index)

    action = None
    "*** YOUR CODE HERE ***"

    if not legalActions:
      return action
    elif len(legalActions) == 1:
      return legalActions[0]
    else:
      if util.flipCoin(self.epsilon):
        return random.choice(legalActions)
      else:
        return self.getPolicy(gameState)

  def doAction(self,state,action):
    """
        Called by inherited class when
        an action is taken in a state
    """
    self.lastState = state
    self.lastAction = action

  def getPolicy(self, gameState):
    return self.computeActionFromQValues(gameState)



  def getValue(self, gameState):
    return self.computeValueFromQValues(gameState)


  def update(self, gameState, action, nextState, reward):
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    oldInvaders = len(invaders)

    features = self.featExtractor.getFeatures(gameState,action, self.index, self)

    qValueCurrent = self.getQValue(gameState,action)
    qValueNext = self.computeValueFromQValues(nextState)

    for feat in features.keys():
      self.weights[feat] = self.weights[feat] + self.alpha * ((reward + self.gamma * (qValueNext)) - qValueCurrent) * features[feat]
    #print(self.weights)

  def observeTransition1(self, state,action,nextState,deltaReward):
    self.episodeRewards += deltaReward
    self.update(state,action,nextState,deltaReward)

  def observationFunction1(self, state):
    if not self.lastState is None:
      reward = state.getScore() - self.lastState.getScore()
      self.observeTransition1(self.lastState, self.lastAction, state, reward)
    return state

  def chooseActionQ(self, gameState):
    start = time.time()

    actions = gameState.getLegalActions(self.index)


    bestAction = self.getAction0(gameState)

    self.observationFunction1(gameState)

    # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
    return bestAction

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index,action)  # Successor basically becomes a STATE and it contains all the info a gameState might have
    pos = successor.getAgentState(self.index).getPosition()  # position for THIS agent
    if pos != nearestPoint( pos):  # this is because apparently the GHOSTS move much faster than normal pacmans, so it creates a dissonance
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

class ExtractorDefensive():
  """
  Returns simple features for a basic reflex Pacman:
  - whether food will be eaten
  - how far away the next food is
  - whether a ghost collision is imminent
  - whether a ghost is one step away
  """
  def closestFood(self,pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist + 1))
    # no food found
    return None

  def getFeatures(self, state, action, index, agent):


    features = util.Counter()

    successor= agent.getSuccessor(state,action)
    myState = successor.getAgentState(agent.index)
    myPos = myState.getPosition()

    features["bias"] = 1.0

    #{'numInvaders': 0, 'onDefense': 0, 'invaderDistance': 0,'#-of-ghosts-1-step-away'}

    #Computes the number of invaders
    enemies = [successor.getAgentState(i) for i in agent.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:

        dists = [agent.getMazeDistance(myPos, a.getPosition()) for a in invaders]
        features['invaderDistance'] = min(dists)
    else:
        enemylist = agent.getOpponents(successor)
        allDistances = successor.getAgentDistances()
        features['invaderDistance'] = min(allDistances)

    #Computes whether we are on defense or not
    features['onDefense'] = 1
    if myState.isPacman:
        features['onDefense'] = 0

    # compute the location of pacman after he takes the action

    x, y =  state.getAgentPosition(index)
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(x + dx), int(y + dy)


    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[state.getAgentState(agent.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    features['foodRatio']=agent.getFoodYouAreDefending(state).count()/agent.myInitialFood

    #posi=successor.getAgentPosition(index)
    if agent.inMyTerritoryIndex(myPos):
      features['inTerritory']=1
    else:
      features['inTerritory']=0

    features.divideAll(10.0)
    return features

