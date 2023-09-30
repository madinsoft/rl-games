from state.common import OnePlayersAbstractGameState, AbstractGameAction
from players.player2048 import State2048


# class Gym2048Move(AbstractGameAction):
#     def __init__(self, x_coordinate, y_coordinate, value):
#         self.x_coordinate = x_coordinate
#         self.y_coordinate = y_coordinate
#         self.value = value

#     def __repr__(self):
#         return "x:{0} y:{1} v:{2}".format(
#             self.x_coordinate,
#             self.y_coordinate,
#             self.value
#         )


class Gym2048GameState(OnePlayersAbstractGameState):

    env = gym.make('2048-v0', width=2, height=2)

    def __init__(self, state, action=None):
        self.state = state
        self.action = action

    @property
    def game_result(self):
        if not self.player.done:
            return None
        return self.player.log_value

    def is_move_legal(self, move):
        return self.player.is_action_legal(move)

    def is_game_over(self):
        return self.player.done

    def move(self, move):
        if not self.is_move_legal(move):
            raise ValueError(f'move {move} on board {self.player.state} is not legal')
        next_state, reward, done, info = self.player.explore(move)
        if next_state == self.player.state:
            raise ValueError(f'move {move} on board next_state {next_state} is the same than previous {self.player.state} ')
        return Gym2048GameState(next_state, move)

    def get_legal_actions(self):
        return self.player.legal_actions

    def __repr__(self):
        return ' '.join([str(i) for i in self.player.board.flat])


# class Gym2048GameState3(OnePlayersAbstractGameState):

#     def __init__(self, state, action=None):
#         self.player = State2048(state)
#         self.action = action

#     @property
#     def game_result(self):
#         if not self.player.done:
#             return None
#         return self.player.log_value

#     def is_move_legal(self, move):
#         return self.player.is_action_legal(move)

#     def is_game_over(self):
#         return self.player.done

#     def move(self, move):
#         if not self.is_move_legal(move):
#             raise ValueError(f'move {move} on board {self.player.state} is not legal')
#         next_state, reward, done, info = self.player.explore(move)
#         if next_state == self.player.state:
#             raise ValueError(f'move {move} on board next_state {next_state} is the same than previous {self.player.state} ')
#         return Gym2048GameState(next_state, move)

#     def get_legal_actions(self):
#         return self.player.legal_actions

#     def __repr__(self):
#         return ' '.join([str(i) for i in self.player.board.flat])




# class Gym2048GameState2(OnePlayersAbstractGameState):

#     def __init__(self, state, player):
#         self.state = state.copy()
#         self.player = player

#     @property
#     def game_result(self):
#         self.player.state = self.state
#         if not self.player.done:
#             return None
#         return self.player.top

#     def is_move_legal(self, move):
#         self.player.state = self.state
#         return self.player.is_action_legal(move)

#     def is_game_over(self):
#         self.player.state = self.state
#         return self.player.done

#     def move(self, move):
#         self.player.state = self.state
#         if not self.is_move_legal(move):
#             raise ValueError(f'move {move} on board {self.state} is not legal')
#         next_state, reward, done, info = self.player.explore(move)
#         return Gym2048GameState(next_state, self.player)

#     def get_legal_actions(self):
#         self.player.state = self.state
#         return self.player.legal_actions

#     def __repr__(self):
#         return ' '.join([str(i) for i in self.state.flat])
