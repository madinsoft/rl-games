from state.common import OnePlayersAbstractGameState


class StateGame2048(OnePlayersAbstractGameState):

    def __init__(self, env, state, action=None):
        self.env = env
        self.state = state
        self.action = action

    @property
    def game_result(self):
        if not self.env.is_done(self.state):
            return None
        return max(self.state)

    def is_move_legal(self, move):
        return self.env.is_action_legal(move, self.state)

    def is_game_over(self):
        return self.env.is_done(self.state)

    def move(self, move):
        if not self.is_move_legal(move):
            raise ValueError(f'move {move} on board {self.state} is not legal')
        next_state, reward, done, info = self.env.explore(move, self.state)
        if (next_state == self.state).all():
            raise ValueError(f'move {move} on board next_state {next_state} is the same than previous {self.state} ')
        return StateGame2048(self.env, next_state, move)

    def get_legal_actions(self):
        return self.env.legal_actions(self.state)

    def __repr__(self):
        return ' '.join([str(i) for i in self.state])
