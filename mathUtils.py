import numpy as np


def getAngle(a, b, c):
  _a = np.array(a)
  _b = np.array(b)
  _c = np.array(c)

  ba = _a - _b
  bc = _c - _b

  cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
  angle = np.arccos(cosine_angle)

  return np.degrees(angle)