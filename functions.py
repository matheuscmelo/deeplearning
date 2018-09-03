import math

def sigmoid(x):
  return 1 / (1.0 + math.exp(-x))

def same(x):
    return x