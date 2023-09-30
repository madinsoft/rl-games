import numpy as np
# from collections import defaultdict
from abc import ABC, abstractmethod


class MctsNodeBase(ABC):

    nid = 0

    def __init__(self, pol, state, parent=None):
        """
        Parameters
        ----------
        state : mctspy.games.common.TwoPlayersAbstractGameState
        parent : MctsNodeBase
        """
        self.pol = pol
        self.state = state
        self.parent = parent
        self.children = []
        MctsNodeBase.nid += 1
        self.id = MctsNodeBase.nid
        if parent:
            self.depth = parent.depth + 1
        else:
            self.depth = 0

    @property
    @abstractmethod
    def untried_actions(self):
        """

        Returns
        -------
        list of mctspy.games.common.AbstractGameAction

        """
        pass

    @property
    @abstractmethod
    def q(self):
        pass

    @property
    @abstractmethod
    def n(self):
        pass

    @abstractmethod
    def expand(self):
        pass

    @abstractmethod
    def is_terminal_node(self):
        pass

    @abstractmethod
    def roll_out(self):
        pass

    @abstractmethod
    def backpropagate(self, reward):
        pass

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        choices_weights = [
            (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]


class MctsNode(MctsNodeBase):

    def __init__(self, pol, state, parent=None):
        super().__init__(pol, state, parent)
        self._number_of_visits = 0.
        self._results = 0
        self._untried_actions = None

    @property
    def untried_actions(self):
        if self._untried_actions is None:
            self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    @property
    def q(self):
        return self._results

    @property
    def n(self):
        return int(self._number_of_visits)

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.move(action)
        child_node = MctsNode(self.pol, next_state, parent=self)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    def roll_out(self):
        current_rollout_state = self.state
        count = 0
        while not current_rollout_state.is_game_over():
            # possible_moves = current_rollout_state.get_legal_actions()
            # if len(possible_moves) == 0:
            #     break
            # action = self.rollout_policy(possible_moves)
            action = self.pol(current_rollout_state.state)
            current_rollout_state = current_rollout_state.move(action)
            count += 1
        return current_rollout_state.game_result

    def backpropagate(self, result):
        self._number_of_visits += 1.
        # self._results[result] += 1.
        self._results += result
        # print('backpropagate', self)
        if self.parent:
            self.parent.backpropagate(result)

    def __repr__(self):
        p = int(self.q / self.n * 100) / 100
        return f'id {self.id} d {self.depth} s |{self.state}| {self.q}/{self.n} {p}%'
