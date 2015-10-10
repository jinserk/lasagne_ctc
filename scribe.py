import pickle
import numpy as np
import sys
from digit import digit_chars

def print_slab(slab):
    """
    Prints a 'slab' of printed 'text' using ascii.
    :param slab: A matrix of floats from [0, 1]
    """
    for ir, r in enumerate(slab):
        print('{:2d}¦'.format(ir), end='')
        for val in r:
            if   val < 0.0:  print('-', end='')
            elif val < .15:  print(' ', end=''),
            elif val < .35:  print('░', end=''),
            elif val < .65:  print('▒', end=''),
            elif val < .85:  print('▓', end=''),
            elif val <= 1.:  print('█', end=''),
            else:            print('+', end='')
        print('¦')

def prepare_print_pred(num_classes):

    def print_pred(y_hat, ignore_repeat=False):
        blank_symbol = num_classes
        res = []
        for i, s in enumerate(y_hat):
            if (s != blank_symbol) and (ignore_repeat or ((i == 0 or s != y_hat[i - 1]))):
                res += [s]
        if len(res) > 0:
            return " ".join(map(str, list(res)))
        else:
            return "-"# * target_seq_len

    return print_pred

class Scribe():
    def __init__(self, alphabet, avg_seq_len, noise=0., vbuffer=2, hbuffer=3,):
        self.alphabet = alphabet
        self.len = avg_seq_len
        self.hbuffer = hbuffer
        self.vbuffer = vbuffer
        self.nDims = alphabet.maxHt + vbuffer
        self.noise = noise

    def get_sample_length(self, vary):
        return self.len + vary * (np.random.randint(self.len // 2)
                                  - self.len // 4)

    def get_sample(self, vary):
        length = self.get_sample_length(vary)
        ret_x = np.zeros((self.nDims, length), dtype=float)
        ret_y = []

        ix = np.random.exponential(self.hbuffer) + self.hbuffer
        while ix < length - self.hbuffer - self.alphabet.maxWd:
            index, char, bitmap = self.alphabet.random_char()
            ht, wd = bitmap.shape
            at_ht = np.random.randint(self.vbuffer +
                                      self.alphabet.maxHt - ht + 1)
            ret_x[at_ht:at_ht+ht, ix:ix+wd] += bitmap
            ret_y += [index]
            ix += wd + np.random.randint(self.hbuffer+1)

        ret_x += self.noise * np.random.normal(size=ret_x.shape,)
        ret_x = np.clip(ret_x, 0, 1)
        return ret_x, ret_y


if __name__ == '__main__':
    alphabet_name = "digit"
    avg_seq_len = 30
    noise = 0.05
    variable_len = True
    out_file_name = 'digit.pkl'

    scribe = Scribe(digit_chars, avg_seq_len, noise)

    xs = []
    ys = []
    for i in range(10000):
        x, y = scribe.get_sample(variable_len)
        xs.append(x)
        ys.append(y)
        print(y, "".join(digit_chars.chars[i] for i in y))
        print_slab(x)

    print('Output: {}\n'
          'Char set : {}\n'
          '(Avg.) Len: {}\n'
          'Varying Length: {}\n'
          'Noise Level: {}'.format(
        out_file_name, digit_chars.chars, avg_seq_len, variable_len, noise))

    with open(out_file_name, 'wb') as f:
        pickle.dump({'x': xs, 'y': ys, 'chars': digit_chars.chars}, f, -1)

