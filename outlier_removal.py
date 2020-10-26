import numpy as np
from scipy.stats import zscore
from scipy import interpolate

from load_dlc_data import *
from preprocessing import running_mean

class outlier_removed_labelarray():

    def __init__(self, df, label, thresh=3, movementframes=None):

        def fill_nan(array):
            """
            interpolate the NAN values of an array
            :param array: 1D array
            :return: 1D array with interpolated former NAN values
            """
            idx = np.arange(array.shape[0])
            good = np.where(np.isfinite(array))
            interp = interpolate.interp1d(idx[good], array[good], bounds_error=False)
            return np.where(np.isfinite(array), array, interp(idx))

        def cleanup_data(df, bodypart, thresh=3, movementframes=None):
            """
            Outlier analysis based on values larger than thresh times the standard deviation
            -----------------------------------------------------------------------------------
            by default thresh=3. The outliers are converted to NAN values and filled using interpolation.
            using the movementframes, you can selectively apply this only to a specific region

            :param df: dlc dataframe
            :param bodypart: dlc bodypart tracked in string
            :param movementframes: list of dim=2 where [idx of movement start frame, idx of movement end frame]
            :param thresh: thresh*sigma, standard dev
            :return:
            """
            if movementframes == None:
                xdata, ydata = np.array(bodypart_array(dataframe_per_bodypart(df, bodypart), pos='x')), \
                               np.array(bodypart_array(dataframe_per_bodypart(df, bodypart), pos='y'))
            else:
                xdata, ydata = np.array(bodypart_array(dataframe_per_bodypart(df, bodypart), pos='x'))[
                               movementframes[0]:movementframes[1]], \
                               np.array(bodypart_array(dataframe_per_bodypart(df, bodypart), pos='y'))[
                               movementframes[0]:movementframes[1]]  # reject values thresh*SIGMA standarddev
            outlier_x = (np.abs(zscore(xdata - running_mean(x=xdata, N=25), nan_policy='omit')) > thresh)
            outlier_y = (np.abs(zscore(ydata - running_mean(x=ydata, N=25), nan_policy='omit')) > thresh)
            outlier_indexes = np.any([outlier_x, outlier_y], axis=0)  # OR operator
            xdata[outlier_indexes], ydata[outlier_indexes] = np.nan, np.nan
            return fill_nan(xdata), fill_nan(ydata)

        self.x, self.y = cleanup_data(df=df, bodypart=label, thresh=thresh, movementframes=None)

if __name__ == "__main__":
    # import data
    df = dlc_data_to_dataframe("/home/sytjon/PycharmProjects/python/DLC/demodata/2182recid_bodycam_fiji50fpsDLC_resnet50_wheelbottomviewJan7shuffle1_1030000.h5")

    # retrieve outlier removed array
    LFarray = outlier_removed_labelarray(df=df, label="left_front", thresh=3, movementframes=[54305,54800])

    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    # load video frame to visualise if the removed points make sense
    img = mpimg.imread("/home/sytjon/PycharmProjects/python/DLC/demodata/2182recid_bodycam_Wheel.png")
    plt.imshow(img)
    # plot the array before outlier removal as a test it worked
    plt.scatter(np.array(bodypart_array(dataframe_per_bodypart(df, "left_front"), pos='x'))[54305:54800],
                np.array(bodypart_array(dataframe_per_bodypart(df, "left_front"), pos='y'))[54305:54800],
                alpha=0.5, color="tab:red")
    plt.scatter(LFarray.x, LFarray.y, alpha=0.05, color="tab:green")
    plt.show()

