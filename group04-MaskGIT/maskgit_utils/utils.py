import numpy as np


def cosine_scheduler(r):
	return np.cos(r * np.pi / 2)