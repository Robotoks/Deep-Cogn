

import numpy as np


def Distance(x, y):
	return np.linalg.norm(np.array(x[:2]) - np.array(y[:2]))


def ManhattanDistance(x, y):
	return np.sum(np.abs(np.array(x[:2]) - np.array(y[:2])))