from abc import ABC, abstractmethod
import numpy as np
from numpy import array as A


# ____________________________________________________________ BaseTransformer
class BaseTransformer(ABC):

    # ________________________________________________ abstractmethod
    @abstractmethod
    def encode_value(self, value):
        pass

    @abstractmethod
    def decode_value(self, value):
        pass

    @abstractmethod
    def _round(self, values):
        pass

    @abstractmethod
    def length(self):
        pass

    # ________________________________________________ methods
    def encode(self, values):
        try:
            return self.encode_value(values)
        except TypeError:
            try:
                return self.encode_list(values)
            except TypeError:
                return self.encode_table(values)

    def __call__(self, values):
        return self.encode(values)

    def decode(self, values):
        try:
            return self.decode_table(values)
        except TypeError:
            try:
                return self.decode_list(values)
            except TypeError:
                return self.decode_value(values)

    def encode_list(self, values):
        return [self.encode_value(value) for value in values]

    def encode_table(self, table):
        return [self.encode_list(liste) for liste in table]

    def decode_list(self, values):
        return [self.decode_value(value) for value in values]

    def decode_table(self, table):
        return [self.decode_list(liste) for liste in table]


# ____________________________________________________________ RangeTransfomer
class RangeTransfomer(BaseTransformer):
    """
    Transformer a list of values in a range values between [.2, .8] well adapted to sigmoid
    [0, 1, 2, 3] => [0.2, 0.4, 0.6, 0.8]
    """
    def __init__(self, values):
        self.min = min(values)
        self.max = max(values)
        self.diam = (self.max - self.min) / .6
        self.possible_values = A(values)

    # ________________________________________________ abstractmethod
    def encode_value(self, value):
        return (value - self.min) / self.diam + .2

    def decode_value(self, value):
        return self._round((value - .2) * self.diam + self.min)

    def _round(self, value):
        distances = (self.possible_values - value)**2
        closest = distances.argmin()
        return self.possible_values[closest]

    def encode(self, values):
        if hasattr(values, '__iter__'):
            return [(value - self.min) / self.diam + .2 for value in values]
        return (values - self.min) / self.diam + .2

    @property
    def length(self):
        return len(self.possible_values)


# ____________________________________________________________ Transformer
class LabelTransformer(BaseTransformer):
    """
    Transformer label/value in hot one format:
    - labels [0, 1, 2, 3] => [(1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)]
    """

    def __init__(self, labels):
        self.table = {}
        self.inv_table = {}
        nb = len(labels)
        ones = np.zeros(nb)
        ones[-1] = 1
        self._length = 0
        for label in labels:
            ones = np.roll(ones, 1)
            tones = tuple(ones)
            self.table[label] = tones
            self.inv_table[tones] = label
            self._length = len(tones)

    # ________________________________________________ abstractmethod
    def encode_value(self, value):
        try:
            return self.table[value]
        except KeyError:
            print(f'value {value}')
            for key, value in self.table.items():
                print(f'key: {key} {value}')
            raise

    def decode_value(self, value):
        try:
            return self.inv_table[self._round(value)]
        except KeyError:
            print('value', value)
            print('inv table', self.inv_table)
            raise

    def _round(self, values):
        i = A(values).argmax()
        ones = np.zeros(len(values))
        ones[i] = 1
        return tuple(ones)

    @property
    def length(self):
        return self._length


# ____________________________________________________________ Transformer
class LabelTransformerFlat(LabelTransformer):
    """
    Transformer label/value in hot one format and convert to a flat list:
    - labels [0, 1, 2, 3] => [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    """

    def encode_list(self, values):
        res = []
        for value in values:
            res += self.encode_value(value)
        return res


# ____________________________________________________________ Transformer
class LabelTransformerOut(LabelTransformer):

    def __call__(self, values):
        return values


# =================================================================
if __name__ == '__main__':
    state_values = [0, 1, 2, 3, 4]
    action_values = [0, 1, 2, 3]
    # ____________________________________________________________
    range_trans = RangeTransfomer(action_values)
    code = range_trans(action_values)
    print(f'range transformer {code}')
    # ____________________________________________________________
    simple_trans = LabelTransformer(action_values)
    code = simple_trans(action_values)
    print(f'Label transformer {code}')
    # ____________________________________________________________
    state_trans = LabelTransformerFlat(state_values)
    code = state_trans(action_values)
    print(f'Label transformer flat {code}')
    # ____________________________________________________________
    action_trans = LabelTransformerOut(action_values)
    code = action_trans(action_values)
    print(f'Label transformer out {code}')
    # ____________________________________________________________
    code = state_trans(2)
    decode = state_trans.decode(code)
    print('state code 2', code, decode)
    # ____________________________________________________________
    code = state_trans(state_values)
    print('state code ', state_values, code)
    code = (.1, .2, 0, .1)
    decode = action_trans.decode(code)
    print('actions code', action_trans(code))
    print('actions decode', code, decode)

    # ____________________________________________________________
    values = A([A([1, 2, 3, 4]), A([0, 0, 0, 0])])
    code = state_trans(values)
    print('code table')
    for line in code:
        print(line)
    code = [(.1, .2, 0, .1), (.2, .8, .1, .05)]
    decode = action_trans.decode(code)
    print('decode table')
    for line in decode:
        print(line)

