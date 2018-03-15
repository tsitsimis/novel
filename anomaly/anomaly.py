import numpy as np
from lemonpy import utils


class SimpleLemon:
    def __init__(self, lemon1, lemon2):
        self.lemon1 = lemon1
        self.lemon2 = lemon2

    def make_lemons(self, pie):
        return (self.lemon1 + self.lemon2) * pie
