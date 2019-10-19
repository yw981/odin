import numpy as np

RESULT_DIR = '../result'

def print5(tag, arr):
    print(tag, arr[0:5, ...])


def arr_stat(tag, arr):
    print(tag + " count ", arr.shape[0], " max ", np.max(arr), " min ", np.min(arr), " mean ", np.mean(arr), " var ",
          np.var(arr), " median ", np.median(arr))


if __name__ == "__main__":
    print()
