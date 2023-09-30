from policies.abstract_policy import AbstractPolicy
from mcts.nodes import MctsNode
from mcts.search import MonteCarloTreeSearch
from state.state_game_2048 import StateGame2048


# ________________________________________________________________
class PolicyMcts(AbstractPolicy):

    def __init__(self, pol, env, nb=100, greedy=0):
        super().__init__(env, greedy=greedy)
        self.pol = pol
        self.nb = nb

    def _action(self, state):
        """ action to do from a specific policy """
        game_state = StateGame2048(self.env, state)
        root = MctsNode(self.pol, game_state)
        mcts = MonteCarloTreeSearch(root)
        best_node = mcts.best_action(self.nb)
        return best_node.state.action

    def __call__(self, state):
        return self.action(state)
