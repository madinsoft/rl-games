import site
site.addsitedir('/home/patrick/projects/IA/my-2048')
from time import perf_counter
from mcts.nodes import OnePLayerGameMonteCarloTreeSearchNode
from mcts.search import MonteCarloTreeSearch
# from mcts.graph import MonteCarloTreeSearchGraph
from state.gym2048 import Gym2048GameState
from players.player2048 import State2048

env = State2048.env
env.seed(1)
# for size in [10, 100, 1000]:
size = 100
cronos = perf_counter()
env.state = [0, 1, 1, 0]
print(size, env.state)
initial_board_state = Gym2048GameState(state=env.board)
root = OnePLayerGameMonteCarloTreeSearchNode(state=initial_board_state)
mcts = MonteCarloTreeSearch(root)
best_node = mcts.best_action(size)
elapsed = perf_counter() - cronos
print(best_node.state.action, f'elapsed {elapsed:.2f} seconds', best_node)

# grapher = MonteCarloTreeSearchGraph(root)
# grapher.build()
# grapher.dump()
