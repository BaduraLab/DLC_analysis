from relative_to_bodyaxis import pair_angle

def angle_between_labels(labelA, labelB):
    Ax, Ay = labelA.x, labelA.y
    Bx, By = labelB.x, labelB.y

    theta = []
    for i_0 in range(len(Ax)):
        theta.append(pair_angle([Ax[i_0], Ay[i_0]],
                                [Bx[i_0], By[i_0]]))
    return np.array(theta)

if __name__ == "__main__":
    import numpy as np
    import math

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


    nose = label_array(df, "nose")
    tail = label_array(df, "T1")

    theta = angle_between_labels(labelA=nose,
                                 labelB=tail)

    time = frames_array(df)

    d_thetha = np.diff(theta)/np.diff(time)
    """
    line plot of the body swing
    """
    import matplotlib.pyplot as plt

    plt.plot(frames_array(df), theta, label = r"$\theta$")
    plt.plot(frames_array(df)[1:], d_thetha, label = r"$d\theta$")

    plt.xlabel("frames")
    plt.ylabel(r"$\theta$(rad)")
    plt.legend()
    plt.show()

    """
    polar plot of the body swing over time 
    """
    #for the polar plots the theta is better visualised in degrees
    d_thetha = [math.degrees(angle) for angle in d_thetha]

    # polar plot config
    fig = plt.figure(tight_layout=True)
    ax_polar = fig.add_subplot(111, polar=True)
    ax_polar.set_thetamin(-90), ax_polar.set_thetamax(90) # show only upper half of the plot
    ax_polar.set_theta_zero_location('E', offset=90) # in video the body axis is horizontal, to make it vertical
    ax_polar.set_title(r"Bodyaxis orientation ($d\theta$, radii=frames)") # title
    # polar information in degrees
    ax_polar.plot(d_thetha, time[1:], color="pink", alpha=.7)

    plt.show()

    """
    polar barplot of the mean body swing 
    --------------------------------------
    width = standard deviation of the bodyangle change within trial
    line = mean bodyaxis deviation
    """
    fig = plt.figure(tight_layout=True)
    ax_polar = fig.add_subplot(111, polar=True)
    ax_polar.set_thetamin(-90), ax_polar.set_thetamax(90)  # show only upper half of the plot
    ax_polar.set_theta_zero_location('E', offset=90)  # in video the body axis is horizontal, to make it vertical
    ax_polar.set_title(r"Mean Bodyaxis Orientation ($d\theta$, radii=frames)") # title
    # polar information in degrees
    ax_polar.bar(np.mean(d_thetha), time[1:], width=np.std(d_thetha) , color="pink", alpha=.7)
    ax_polar.plot(np.mean(d_thetha)*np.ones_like(d_thetha), time[1:], color="tab:red", alpha=.7, label="Mouse 1")

    plt.legend()
    plt.show()