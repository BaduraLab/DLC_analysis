import numpy as np
import pandas as pd

def dlc_data_to_dataframe(file):
    """
    Import position label data from deeplabcut into pandas dataframe
    ---------------------------------------------------------
    :param file: dlc position data of the labels (hdf5)
    :return: pandas data frame with label positions
    """

    df = pd.read_hdf (file)
    df.drop ('likelihood', axis=1, level='coords', inplace=True) # remove likelihood measurep
    return df

def bodyparts_list(df, keyword=None):
    """
    List object of all the DLC tracked features/labels
    ---------------------------------------------------------
    There is an option to supply keyword so you can select a subset, for exlcusion similar approach can be taken,
    but is not build in this function

    :param dataframe: dlc dataframe with label positions
    :param keyword: return labels only when speicified keyword is in the label name, e.g. "front" or "L"
    :return: list of dlc labels
    """

    labels = list (set (list (df.columns.get_level_values ('bodyparts'))))

    if keyword is not None:
        labels = [label for label in labels if keyword in label]

    return labels

def frames_array(df, fps=None):
    """
    Create time vector (s) based on fps
    ----------------------------------
    :param dataframe:
    :param fps: frames per second from video acquisition
    :return:
    """
    time = 1
    if isinstance(fps, int): # check if inputted fps is integer
        time=fps
    return np.array (list (df.index))/time

def dataframe_per_bodypart(df, bodypart):
    """
    Dataframe per bodypart
    -------------------------
    :param df:
    :param bodypart: string of bodypart dlc label
    :return:
    """
    scorer = list (df.columns.get_level_values ('scorer'))[0]
    df_bodypart = df.xs (bodypart, level='bodyparts', axis=1)
    df_bodypart = df_bodypart.xs (scorer, level='scorer', axis=1)
    return df_bodypart

def bodypart_array(df_bodypart, pos='x'):
    return np.array (df_bodypart[pos].values)


if __name__ == "__main__":
    """
    example use case
    - import data
    - get raw traces data into numpy array only from selective features (left side of animal)
    - plot x position over time
    """
    import matplotlib.pyplot as plt

    # you can initiate your label array here
    # This way if you want to do common operations on your array (here, only raw is applied) it is all done here, looks a bit more orderly
    # for example, if you want do do smoothing or outlier removal etc.
    class raw_label_array:
        """
        Raw DLC label (specfied by name) coordinates
        """

        def __init__(self, df, label):
            # raw video x,y pixel coordinates
            self.x = np.array(dataframe_per_bodypart(df, label)["x"].values)
            self.y = np.array(dataframe_per_bodypart(df, label)["y"].values)

    # import DLC data to table/dataframe
    df = dlc_data_to_dataframe("demodata/16454_07_42DLC_resnet50_BaduraLocomousejul22shuffle1_200000.h5")

    # side view only labels
    labels = bodyparts_list(df=df,
                            keyword="L"
                            )

    # extract time information
    time = frames_array(df, fps=50)

    # perform extraction of raw coordinates over every label
    for label in labels:
        print(label)

        array = raw_label_array(df=df, label=label)
        print("x-coordinates array dim", array.x.shape)
        print("y-coordinates array dim", array.y.shape)

        # plot y-information over time in scatterplot
        plt.plot(time, array.x, label=label)
    plt.legend()
    plt.show()