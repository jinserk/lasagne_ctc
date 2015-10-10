"""*********************************************************************
 http://opengameart.org/content/8x8-ascii-bitmap-font-with-c-source
 Copyright (C) Lisa Milne 2014 <lisa@ltmnet.com>

*********************************************************************"""
import numpy as np

# the values in this array are a 8x8 bitmap font for ascii characters
bitmaps_hex = [
    0x1824424224180000,  # 0 
    0x8180808081C0000,   # 1 
    0x3C420418207E0000,  # 2 
    0x3C420418423C0000,  # 3 
    0x81828487C080000,   # 4 
    0x7E407C02423C0000,  # 5 
    0x3C407C42423C0000,  # 6 
    0x7E04081020400000,  # 7 
    0x3C423C42423C0000,  # 8 
    0x3C42423E023C0000,  # 9 
]

bitmaps = np.array(bitmaps_hex, dtype=np.uint64)
bitmaps = np.unpackbits(bitmaps.view(np.uint8))
bitmaps = bitmaps.reshape(len(bitmaps_hex), 8, 8)
bitmaps = bitmaps[:,::-1,:]
chars = [chr(x) for x in range(48, 10)]

class Character():
    def __init__(self, bitmaps, chars=None):
        self.bitmaps = bitmaps
        self.n = len(bitmaps)
        self.chars = chars if chars else [str(i) for i in range(len(bitmaps))]

        self.maxHt = max([bitmap.shape[0] for bitmap in bitmaps])
        self.maxWd = max([bitmap.shape[1] for bitmap in bitmaps])

    def random_char(self):
        index = np.random.choice(self.n)
        bitmap = self.bitmaps[index]
        char = self.chars[index]
        return index, char, bitmap

    def __str__(self):
        ret = ""
        for c, b in zip(self.chars, self.bitmaps):
            slab = "\n".join(("".join("# "[p] for p in r) for r in b))
            ret += "\n{}:\n{}".format(c, slab)
        return ret

digit_chars = Character(bitmaps, chars)

