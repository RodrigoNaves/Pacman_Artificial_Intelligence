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


from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
import capture
from game import Actions
import sys

####PIECE OF CODE TO COUNT HOW LONG AN ALGORITHM TAKES TO RUN
start = time.time()
# insert code

# print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

import numpy as np

#################
# Team creation #
#################

arguments = {}
sys.path.append("teams/<AImirant>/")


def createTeam(firstIndex, secondIndex, isRed,
               first='offensiveAgent', second='DefensiveAgent', numTraining=0):

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


    def registerInitialState(self, gameState):
        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''

        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

        if self.red:
            self.edge = (gameState.data.layout.width - 2) // 2
        else:
            self.edge = ((gameState.data.layout.width - 2) // 2) + 1
        self.boundary = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(self.edge, i):
                self.boundary.append((self.edge, i))



class offensiveAgent(AIagent):
    def __init__(self, index):
        AIagent.__init__(self, index)

        self.weights = util.Counter()
        self.features = util.Counter()
        self.edge = None
        self.weights = {'bias': -1.7240061402858196, '#-of-ghosts-1-step-away': -35.674089224737658, 'eats-food': 8.094756913804474, 'closest-food': -0.09221658624921276, 'carryingFood': 30}
        self.boundary = []

        self.lastState = None

    def registerInitialState(self, gameState):
     '''
     Make sure you do not delete the following line. If you would like to
     use Manhattan distances instead of maze distances in order to save
     on initialization time, please take a look at
     CaptureAgent.registerInitialState in captureAgents.py.
     '''

     self.start = gameState.getAgentPosition(self.index)
     CaptureAgent.registerInitialState(self, gameState)


     if self.red:
         self.edge = ((gameState.data.layout.width - 2) // 2)- 1
     else:
         self.edge = ((gameState.data.layout.width - 2) // 2) + 2
     self.boundary = []
     for i in range(1, gameState.data.layout.height - 1):
         if not gameState.hasWall(self.edge, i):
             self.boundary.append((self.edge, i))

     self.q = util.Counter()
     self.featExtractor = Extractor()
     self.startEpisode()


    def startEpisode(self):
        """
          Called by environment when new episode is starting
        """
        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0

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
            if len(set(count)) == 1:
                return (list(count.keys()))[0][1]
            return count.argMax()[1]


    def getAction0(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
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

        features = self.featExtractor.getFeatures(gameState,action, self.index, self)

        qValueCurrent = self.getQValue(gameState,action)
        qValueNext = self.computeValueFromQValues(nextState)

        for feat in features.keys():
            self.weights[feat] = self.weights[feat] + self.alpha * ((reward + self.gamma * (qValueNext)) - qValueCurrent) * features[feat]
        #print(self.weights)
        
    def observeTransition1(self, state,action,nextState,deltaReward):
        """
            Called by environment to inform agent that a transition has
            been observed. This will result in a call to self.update
            on the same arguments

            NOTE: Do *not* override or call this function
        """
        self.episodeRewards += deltaReward
        self.update(state,action,nextState,deltaReward)

    def observationFunction1(self, state):
        """
            This is where we ended up after our last action.
            The simulation should somehow ensure this is called
        """


        if not self.lastState is None:
            reward = state.getScore() - self.lastState.getScore()
            self.observeTransition1(self.lastState, self.lastAction, state, reward)
        return state

    def chooseAction(self, gameState):
        start = time.time()

        actions = gameState.getLegalActions(self.index)


        bestAction = self.getAction0(gameState)

        self.observationFunction1(gameState)
        foodLeft = len(self.getFood(gameState).asList())
        if foodLeft <= 2:
            i = 2  # we use this counter to increase the importance of returning home when we know we have elapsed more turns
            i += 1
            if foodLeft <= i:
                bestDist = 9999
                for action in actions:
                    successor = self.getSuccessor(gameState, action)
                    pos2 = successor.getAgentPosition(self.index)
                    dist = self.getMazeDistance(self.start, pos2)
                    if dist < bestDist:
                        bestAction = action
                        bestDist = dist
                return bestAction

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
class DefensiveAgent(AIagent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def __init__(self, index):
        AIagent.__init__(self, index)

        self.weights = util.Counter()
        self.features = util.Counter()
        self.edge = None
        self.weights = {'bias': 114.81860095588428, 'numInvaders': 6.222840458134068, 'onDefense': 117.82383168359877, 'invaderDistance': 1.6049848124918615, 'stop': 86.02286730224952, 'reverse': -39.20985043044917, 'foodRatio': 77.92207826189737, 'inTerritory': -385.18139904411584}
        self.boundary = []
        self.lastState = None
        self.myInitialFood=0

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


     if self.red:
         self.edge = ((gameState.data.layout.width - 2) // 2)- 1
     else:
         self.edge = ((gameState.data.layout.width - 2) // 2) + 2
     self.boundary = []
     for i in range(1, gameState.data.layout.height - 1):
         if not gameState.hasWall(self.edge, i):
             self.boundary.append((self.edge, i))
   

     self.q = util.Counter()
     self.featExtractor = ExtractorDefensive()
     self.startEpisode()


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
            if len(set(count)) == 1:
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

        features = self.featExtractor.getFeatures(gameState,action, self.index, self)

        qValueCurrent = self.getQValue(gameState,action)
        qValueNext = self.computeValueFromQValues(nextState)

        for feat in features.keys():
            self.weights[feat] = self.weights[feat] + self.alpha * ((reward + self.gamma * (qValueNext)) - qValueCurrent) * features[feat]
        
    def observeTransition1(self, state,action,nextState,deltaReward):

        self.episodeRewards += deltaReward
        self.update(state,action,nextState,deltaReward)

    def observationFunction1(self, state):

        if not self.lastState is None:
            reward = state.getScore() - self.lastState.getScore()
            self.observeTransition1(self.lastState, self.lastAction, state, reward)
        return state

    def chooseAction(self, gameState):
        start = time.time()

        actions = gameState.getLegalActions(self.index)


        bestAction = self.getAction0(gameState)
        self.observationFunction1(gameState)

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
class Extractor():
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
        # extract the grid of food and wall locations and get the ghost locations
        red = state.isOnRedTeam(index)
        if red:
            food = state.getBlueFood()
        else:
            food = state.getRedFood()
        walls = state.getWalls()
        ghosts1 = agent.getOpponents(state)
        ghostlist =[]
        for g in ghosts1:
            if state.getAgentPosition(g) != None:
                ghostlist.append(state.getAgentPosition(g))

        ghosts = ghostlist

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action

        x, y =  state.getAgentPosition(index)
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = self.closestFood((next_x, next_y), food, walls)

        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)

        carryingFood = state.getAgentState(index).numCarrying
        depositFood = 0
        if carryingFood > 3:
            depositFood = carryingFood * 1/ agent.getMazeDistance((x, y), agent.start)#bound) for bound in agent.boundary])

        features['carryingFood'] = depositFood

        features.divideAll(10.0)

        return features


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
        # extract the grid of food and wall locations and get the ghost locations
        features = util.Counter()

        successor= agent.getSuccessor(state,action)
        myState = successor.getAgentState(agent.index)
        myPos = myState.getPosition()

        features["bias"] = 1.0

        #Computes the number of invaders
        enemies = [successor.getAgentState(i) for i in agent.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:

            dists = [agent.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)
        else:
            #enemylist = agent.getOpponents(successor)
            allDistances = successor.getAgentDistances()
            features['invaderDistance'] = min(allDistances)

        #Computes whether we are on defense or not
        features['onDefense'] = 1
        if myState.isPacman:
            features['onDefense'] = 0

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[state.getAgentState(agent.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        features['foodRatio']=agent.getFoodYouAreDefending(state).count()/agent.myInitialFood

        #Inicates whether the agent in its own territory
        if agent.inMyTerritoryIndex(myPos):
            features['inTerritory']=1
        else:
            features['inTerritory']=0


        features.divideAll(10.0)
        return features




