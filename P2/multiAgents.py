# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # return successorGameState.getScore()
        score = successorGameState.getScore()
        foods = newFood.asList()
        f_distance = list()
        for i in foods:
            f_distance.append(manhattanDistance(newPos, i))

        if action == "Stop":
            score -= 200

        for i in range(len(newGhostStates)):
            if (util.manhattanDistance(newGhostStates[i].getPosition(), newPos) < 2) or (
                    newGhostStates[i].getPosition() == newPos and (newScaredTimes[i] == 0)):
                score -= 200
            
        if len(f_distance) > 0:
            min_f_distance = min(f_distance)
            score = score + (1 / min_f_distance)

        return score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """


    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        ghostsNumbers = gameState.getNumAgents()

        def max_value_f(gameState, depth):
            max_value = float('-inf')
            depth -= 1

            if depth == 0:
                return self.evaluationFunction(gameState)

            if gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)

            for a in gameState.getLegalActions(0):
                max_value = max(max_value, min_value_f(gameState.generateSuccessor(0, a), depth, 1))

            return max_value

        def min_value_f(gameState, depth, agentIndex):
            min_value = float('inf')

            if gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)

            for a in gameState.getLegalActions(agentIndex):
                if agentIndex < (ghostsNumbers - 1):
                    min_value = min(min_value,
                                    min_value_f(gameState.generateSuccessor(agentIndex, a), depth, agentIndex + 1))
                else:
                    min_value = min(min_value, max_value_f(gameState.generateSuccessor(agentIndex, a), depth))

            return min_value

        max_value = float('-inf')
        result = 0

        for a in gameState.getLegalActions(0):
            action_value = min_value_f(gameState.generateSuccessor(0, a), self.depth, 1)

            if action_value > max_value:
                max_value = action_value
                result = a

        return result


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        ghostsNumbers = gameState.getNumAgents()

        def max_value_f(gameState, depth, Alpha, Beta):
            depth = depth - 1
            max_value = float('-inf')

            if depth == 0:
                return self.evaluationFunction(gameState)

            if gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)

            for a in gameState.getLegalActions(0):
                max_value = max(max_value, min_value_f(gameState.generateSuccessor(0, a), depth, 1, Alpha, Beta))
                if max_value > Beta:
                    return max_value

                Alpha = max(Alpha, max_value)

            return max_value

        def min_value_f(gameState, depth, agentIndex, Alpha, Beta):
            min_value = float('inf')  # minValue

            if gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)

            for a in gameState.getLegalActions(agentIndex):
                if agentIndex < (ghostsNumbers - 1):
                    min_value = min(min_value, min_value_f(gameState.generateSuccessor(agentIndex, a), depth,
                                                           agentIndex + 1, Alpha, Beta))
                else:
                    min_value = min(min_value,
                                    max_value_f(gameState.generateSuccessor(agentIndex, a), depth, Alpha, Beta))
                if min_value < Alpha:
                    return min_value

                Beta = min(Beta, min_value)

            return min_value

        # main part
        result = 0
        Alpha = float('-inf')
        Beta = float('inf')

        for a in gameState.getLegalActions(0):
            action_value = min_value_f(gameState.generateSuccessor(0, a), self.depth, 1, Alpha, Beta)
            if action_value > Alpha:
                result = a
                Alpha = action_value

        return result


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expect_max(gameState, depth, agentcounter):
            if agentcounter >= gameState.getNumAgents():
                agentcounter = 0
                depth += 1

            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                return self.evaluationFunction(gameState)

            elif agentcounter == 0:
                return max_value_f(gameState, depth, agentcounter)

            else:
                return get_expected_value(gameState, depth, agentcounter)

        def max_value_f(gameState, depth, agentcounter):
            max_value = ["", -float("inf")]

            if not gameState.getLegalActions(agentcounter):
                return self.evaluationFunction(gameState)

            for a in gameState.getLegalActions(agentcounter):
                curr = expect_max(gameState.generateSuccessor(agentcounter, a), depth, agentcounter + 1)
                if type(curr) is not list:
                    next_value = curr
                else:
                    next_value = curr[1]

                if next_value > max_value[1]:
                    max_value = [a, next_value]

            return max_value

        def get_expected_value(gameState, depth, agentcounter):
            max_value = ["", 0]

            if not gameState.getLegalActions(agentcounter):
                return self.evaluationFunction(gameState)

            for a in gameState.getLegalActions(agentcounter):
                curr_state = gameState.generateSuccessor(agentcounter, a)
                curr = expect_max(curr_state, depth, agentcounter + 1)
                if type(curr) is list:
                    next_value = curr[1]
                else:
                    next_value = curr

                max_value[0] = a
                max_value[1] += next_value * (1.0 / len(gameState.getLegalActions(agentcounter)))
            return max_value

        return expect_max(gameState, 0, 0)[0]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    foods = []
    ghost_close = 0

    for food in currentGameState.getFood().asList():
        foods.append(manhattanDistance(currentGameState.getPacmanPosition(), food) * -1)
    if not foods:
        foods.append(0)

    if manhattanDistance(currentGameState.getGhostPositions()[0], currentGameState.getPacmanPosition()) == 0 and \
            currentGameState.getGhostStates()[0].scaredTimer == 0:
        ghost_close = -200

    elif currentGameState.getGhostStates()[0].scaredTimer > 0:
        ghost_close = -1 / (
            manhattanDistance(currentGameState.getGhostPositions()[0], currentGameState.getPacmanPosition()))

    return currentGameState.getScore() + ghost_close + len(currentGameState.getCapsules()) + max(foods)



# Abbreviation
better = betterEvaluationFunction
