import numpy as np


def main():
  n = np.load("../images/m141e_diploide.npy", allow_pickle = True).tolist()
  print(n)

main()