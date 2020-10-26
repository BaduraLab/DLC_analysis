from scipy.signal import savgol_filter
import numpy as np

def smooth_signal(label_array):
    """
    Local polynomial fitting based smoothing
    --------------------------------------------

    :param label_array:
    :return:
    """
    #Savitzky–Golay smoothed filter
    return savgol_filter(label_array, window_length=9, polyorder=2)

def running_mean(x, N):
    """
    Moving average on array
    -------------------------------
    Local average to find the label array with respect to its own center of osscilation for paws

    :param x:
    :param N:
    :return:
    """
    out = np.zeros_like(x, dtype=np.float64)
    dim_len = x.shape[0]
    for i in range(dim_len):
        if N%2 == 0:
            a, b = i - (N-1)//2, i + (N-1)//2 + 2
        else:
            a, b = i - (N-1)//2, i + (N-1)//2 + 1

        #cap indices to min and max indices
        a = max(0, a)
        b = min(dim_len, b)
        out[i] = np.mean(x[a:b])
    return out