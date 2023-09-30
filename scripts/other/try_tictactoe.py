import numpy as np
from mcts.nodes import TwoPlayersGameMonteCarloTreeSearchNode
from mcts.search import MonteCarloTreeSearch
from mcts.graph import MonteCarloTreeSearchGraph
from state.tictactoe import TicTacToeGameState

state = np.zeros((3, 3))

initial_board_state = TicTacToeGameState(state=state, next_to_move=1)
root = TwoPlayersGameMonteCarloTreeSearchNode(state=initial_board_state)
mcts = MonteCarloTreeSearch(root)
best_node = mcts.best_action(10000)

grapher = MonteCarloTreeSearchGraph(root)
grapher.build()
grapher.dump()
