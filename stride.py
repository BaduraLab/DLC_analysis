from scipy.signal import find_peaks

def find_minmax_peaks(array):
    """
    find indexes of where the peaks of interest are occuring
    :param array:
    :return:
    """
    # array -= running_mean(x=array, N=25)

    maxima, _ = find_peaks(array,
                            height=1,
                            distance=20)

    minima, _ = find_peaks(-array,
                            height=1,
                            distance=20)
    return maxima,minima


class stride:
    """
    stride parameters extraction
    ------------------------------
    - min positions
    - max positions
    - stance/stride length
    - stance/stride duration
    """

    def __init__(self, y_label_array):
        """

        :param y_label_array: along body axis information
        """
        # find local maxima and minima for every stride cycle within the array
        maxima, minima = find_minmax_peaks(y_label_array)

        self.minpos = minima
        self.maxpos = maxima

        minmaxpos = np.sort(np.concatenate([self.minpos, self.maxpos]))
        minmax = [True if pos in self.maxpos else False for pos in minmaxpos]  # True for max

        # stance and stride specific information

        swings_len, stances_len = [], []
        swings_dur, stances_dur = [], []
        stancestart = None
        swingstart = None
        for max, pos in zip(minmax, minmaxpos):
            if max and stancestart is None:
                stancestart = pos
            if max is False and stancestart is not None:
                stances_len.append(np.abs(pos - stancestart))
                stances_dur.append(np.abs(y_label_array[int(stancestart)] - y_label_array[int(pos)]))

                stancestart = None

            if max is False and swingstart is None:
                swingstart = pos
            if max and swingstart is not None:
                swings_len.append(np.abs(y[int(swingstart)] - y[int(pos)]))
                swings_dur.append(np.abs(pos - swingstart))

                swingstart = None

        self.stance_length = stances_len
        self.stance_dur = stances_dur
        self.swings_length = swings_len
        self.swings_dur = swings_dur


if __name__ == "__main__":
    from load_dlc_data import *
    from relative_to_bodyaxis import *

    import numpy as np

    class label_array:

        def __init__(self, df, label, smooth=True, COO=False, N=25):
            # raw video x,y pixel coordinates
            xdata = np.array(dataframe_per_bodypart(df, label)["x"].values)
            ydata = np.array(dataframe_per_bodypart(df, label)["y"].values)

            if smooth:
                print(label, "smooth performed")
                from preprocessing import smooth_signal
                xdata = smooth_signal(xdata.copy())
                ydata = smooth_signal(ydata.copy())
            if COO:
                print(label, "COO performed", N)
                from preprocessing import running_mean
                xdata -= running_mean(xdata.copy(), N=N)
                ydata -= running_mean(ydata.copy(), N=N)

            self.x = xdata.copy()
            self.y = ydata.copy()


    """
    import data & preprocessing
    """
    # import DLC data to table/dataframe
    df = dlc_data_to_dataframe("demodata/16454_07_42DLC_resnet50_BaduraLocomousejul22shuffle1_200000.h5")

    nose = label_array(df=df, label="nose")
    tail = label_array(df=df, label="T1")

    R, offset = R_per_frame(xnose_array=nose.x, ynose_array=nose.y,
                            xtail_array=tail.x, ytail_array=tail.y,
                            frames=frames_array(df))

    """
    plot config 
    ------------------------------------------------
    -> smoothing effect on stride parameters extraction
    """
    import matplotlib.pyplot as plt

    fig_stride, axes_stride = plt.subplots(2, 1, sharex=True, sharey=False)
    fig_stridelength, axes_stridelength = plt.subplots(2, 1, sharex=True, sharey=False)
    fig_strideduration, axes_strideduration = plt.subplots(2, 1, sharex=True, sharey=False)
    fig_stridespeed, axes_stridespeed = plt.subplots(2, 1, sharex=True, sharey=False)
    fig_stridescatter, axes_stridescatter = plt.subplots(2, 1, sharex=True, sharey=False)

    fig_stridelength.suptitle("Stride length"), fig_strideduration.suptitle(
        "Stride duration"), fig_stridespeed.suptitle("Stride velocity"), \
    fig_stridescatter.suptitle("Relationship Duration vs. Length of stride")

    axes_stridelength[0].set_title("stance"), axes_stridelength[1].set_title("swing")
    axes_strideduration[0].set_title("stance"), axes_strideduration[1].set_title("swing")
    axes_stridespeed[0].set_title("stance"), axes_stridespeed[1].set_title("swing")
    axes_stridescatter[0].set_title("stance"), axes_stridescatter[1].set_title("swing")

    axes_stridelength[0].set_ylim([0, 100]), axes_stridelength[1].set_ylim([0, 100])
    axes_strideduration[0].set_ylim([0, 70]), axes_strideduration[1].set_ylim([0, 70])
    axes_stridespeed[0].set_ylim([0, 2]), axes_stridespeed[1].set_ylim([0, 2])

    axes_stridescatter[0].set_xlabel("Duration"), axes_stridescatter[1].set_xlabel("Duration")
    axes_stridescatter[0].set_ylabel("length"), axes_stridescatter[1].set_ylabel("length")

    colorlist = {"nose": "pink",
                 "LF": "indianred",
                 "RF": "cornflowerblue",
                 "LB": "darkorange",
                 "RB": "darkorchid",
                 "T1": "gold",
                 "T2": "khaki",
                 "T3": "palegoldenrod",
                 "T4": "darkkhaki",
                 "T5": "olive"}

    """
    plot stride information 
    ------------------------------------------------
    -> smoothing effect on stride parameters extraction
    """
    from preprocessing import running_mean, smooth_signal

    for i0, bodypart in enumerate(["LF", "RF", "LB", "RB"]):
        data = relative_to_body_array(bodypart_array=label_array(df=df, label=bodypart,
                                                                 smooth=False, COO=False, N=25*4),
                                      R=R, offset=offset)

        # non-smoothed based estimation
        y = data.y - running_mean(x=data.y, N=100)
        stridedata = stride(y)

        # plot raw data
        axes_stride[0].plot(y, color=colorlist[bodypart], label=bodypart)
        axes_stride[0].plot(stridedata.maxpos, y[stridedata.maxpos],
                            "x", color=colorlist[bodypart])
        axes_stride[0].plot(stridedata.minpos, y[stridedata.minpos],
                            "x", color=colorlist[bodypart])
        axes_stride[0].plot(np.zeros_like(y), "--", color="gray", alpha=.7)

        # smoothed version
        y = smooth_signal(y)
        stridedata = stride(y)

        axes_stride[1].plot(y, color=colorlist[bodypart], label=bodypart)
        axes_stride[1].plot(stridedata.maxpos, y[stridedata.maxpos],
                            "x", color=colorlist[bodypart])
        axes_stride[1].plot(stridedata.minpos, y[stridedata.minpos],
                            "x", color=colorlist[bodypart])
        axes_stride[1].plot(np.zeros_like(y), "--", color="gray", alpha=.7)

        # stance and swing length plots
        axes_stridelength[0].bar(bodypart, np.mean(stridedata.stance_length), yerr=np.std(stridedata.stance_length, ddof=1),
                                 color=colorlist[bodypart])
        axes_stridelength[1].bar(bodypart, np.mean(stridedata.swings_length), yerr=np.std(stridedata.swings_length, ddof=1),
                                 color=colorlist[bodypart])

        # stance and swing duration plots
        axes_strideduration[0].bar(bodypart, np.mean(stridedata.stance_dur), yerr=np.std(stridedata.stance_dur, ddof=1),
                                   color=colorlist[bodypart])
        axes_strideduration[1].bar(bodypart, np.mean(stridedata.swings_dur), yerr=np.std(stridedata.swings_dur, ddof=1),
                                   color=colorlist[bodypart])

        # scatter plot stance and swing lengths and time relationship
        axes_stridescatter[0].scatter(stridedata.stance_dur, stridedata.stance_length, color=colorlist[bodypart], alpha=.75)
        axes_stridescatter[1].scatter(stridedata.swings_dur, stridedata.swings_length, color=colorlist[bodypart], alpha=.75)

        # barplot speeds of stance and stride
        stancearray_speed = np.divide(stridedata.stance_length, stridedata.stance_dur)
        swingarray_speed = np.divide(stridedata.swings_length, stridedata.swings_dur)

        axes_stridespeed[0].bar(bodypart, np.mean(stancearray_speed), yerr=np.std(stancearray_speed, ddof=1),
                                color=colorlist[bodypart])
        axes_stridespeed[1].bar(bodypart, np.mean(swingarray_speed), yerr=np.std(swingarray_speed, ddof=1),
                                color=colorlist[bodypart])

    plt.show()