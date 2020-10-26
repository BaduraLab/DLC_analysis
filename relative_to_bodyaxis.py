import numpy as np

from math import atan2
def pair_angle(xy_0,xy_1):
    """
    calculate angle of line to x-axis
    --------------------------------------------

    :param xy_0:
    :param xy_1:
    :return: angle between two points in cartesian coordinates
    """
    dx, dy = np.array (xy_1) - np.array (xy_0)
    return atan2(dy,dx) + np.pi

def rotation_matrix(theta):
    """
    Rotate  2D cartesian coordinates by angle
    --------------------------------------------------
    # https://en.wikipedia.org/wiki/Rotation_matrix # explained methodology

    :param theta: angle
    :return: rotaion matrix
    """
    c, s = np.cos (theta), np.sin (theta)
    return np.array (((c, -s), (s, c)))

def R_per_frame(xnose_array: object, ynose_array: object, xtail_array: object, ytail_array: object, frames: object) -> object:
    """
    Rotation per frame
    -------------------------------
    Mouse is moxing, which causes the bodyaxis to move hence a new rotationmatrix is needed per frame
    This function can be used to transform your data based on any "line" defined by two points, whether these are moving points or stationary points.

    :param xnose_array:
    :param ynose_array:
    :param xtail_array:
    :param ytail_array:
    :param frames:
    :return:
    """
    R = []
    offset = []
    for frame in frames:
        xy_nose = [xnose_array[int (frame)], ynose_array[int (frame)]]
        xy_tail = [xtail_array[int (frame)], ytail_array[int (frame)]]

        theta = pair_angle (xy_tail, xy_nose)
        rot_mat =rotation_matrix (-theta - (np.pi / 2))

        xnose, ynose = rot_mat.dot (np.array (xy_nose))
        xtail, ytail = rot_mat.dot (np.array (xy_tail))

        if np.abs (ynose - ytail) < 50:
            R.append (R[-1])
            offset.append (offset[-1])
        else:
            R.append (rot_mat)
            offset.append ([xnose, ynose])
    return R, offset

def relative_to_body_array(bodypart_array, R,offset):
    x_array = []
    y_array = []
    for i_0 in range (len(bodypart_array.x)):
        xprime, yprime = R[i_0].dot(np.array([bodypart_array.x[i_0],bodypart_array.y[i_0]]))
        x_array.append(xprime-offset[i_0][0])
        y_array.append(yprime-offset[i_0][1])

    bodypart_array.x = np.array(x_array)
    bodypart_array.y = np.array(y_array)
    return bodypart_array

from shapely.geometry import Point
from shapely.geometry import LineString
def extract_neck_point(x_nose, y_nose, xy_0, xy_1):
    """
    Find neck point based on headplate coordinates
    -----------------------
    neckpoint is estimated based on supplied nose position
    where the code finds the neckpoint that makes a line with the nose.
    such that said line is perpendicular to the "headplate-line"

    :param x_nose: x position of nose, can be singular point or array
    :param y_nose: y position of nose, can be singular point or array
    :param xy_0: [x, y] where x is the xposition of headplate point, and y is the y position on the headplate point
    :param xy_1: [x, y] where x is the xposition of headplate point, and y is the y position on the headplate point
    :return: neckpoint, singular point or array which can be used to make a bodypart array
    """
    nose_pos = Point (np.mean (x_nose), np.mean (y_nose))
    line = LineString ([tuple (xy_0), tuple (xy_1)])

    x = np.array (nose_pos.coords[0])
    u = np.array (line.coords[0])
    v = np.array (line.coords[len (line.coords) - 1])

    n = v - u
    n /= np.linalg.norm (n, 2)
    P = u + n * np.dot (x - u, n)
    return np.array (P)


if __name__ == "__main__":
    from load_dlc_data import dlc_data_to_dataframe, dataframe_per_bodypart, frames_array

    df = dlc_data_to_dataframe("demodata/16454_07_42DLC_resnet50_BaduraLocomousejul22shuffle1_200000.h5")

    class label_array:
        """
        Raw DLC label (specfied by name) coordinates
        """

        def __init__(self, df, label):
            # raw video x,y pixel coordinates
            self.x = np.array(dataframe_per_bodypart(df, label)["x"].values)
            self.y = np.array(dataframe_per_bodypart(df, label)["y"].values)



    nose = label_array(df,"nose")
    tail = label_array(df,"T1")

    R, offset = R_per_frame(xnose_array=nose.x, ynose_array=nose.y,
                            xtail_array=tail.x, ytail_array=tail.y,
                            frames=frames_array(df))

    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set (palette="Paired")

    ax0 = plt.subplot (211)
    ax1 = plt.subplot (212)


    for bodypart in ["nose", "T1", 'LF', 'RF', "RB", "LB"]:
        rawdata = label_array(df,bodypart)

        if bodypart not in ["nose", "T1"]:
            sns.scatterplot (rawdata.x,rawdata.y,
                             alpha=0.4,label=bodypart, ax=ax0)

            data = relative_to_body_array(bodypart_array=rawdata,
                                          R=R, offset=offset)
            sns.scatterplot(data.x, data.y,
                            alpha=0.4, label=bodypart, ax=ax1)
        else:
            sns.scatterplot(rawdata.x, rawdata.y,
                            alpha=0.4, label=bodypart, ax=ax0, color="pink")


    ax0.set_title ('DLC video tracking')
    ax0.set_xlabel ('x-position')
    ax0.set_ylabel ('y-position')

    ax1.set_title ('Relative to bodyaxis (nose to tail base)')
    ax1.set_xlabel ('Relative to Central Body x-position')
    ax1.set_ylabel ('Relative to Central Body x-position')

    plt.legend ()
    plt.show ()