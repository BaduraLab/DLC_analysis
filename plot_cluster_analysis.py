if __name__ == "__main__":
    # import own code
    from load_dlc_data import *
    from relative_to_bodyaxis import *

    # import libraries
    import numpy as np

    # import DLC data from file
    df = dlc_data_to_dataframe("Data/16454_07_42DLC_resnet50_BaduraLocomousejul22shuffle1_200000.h5")

    class label_array:
        """
        DLC array operations
        """

        def __init__(self, df, label):
            # raw video x,y pixel coordinates
            self.x = np.array(dataframe_per_bodypart(df, label)["x"].values)
            self.y = np.array(dataframe_per_bodypart(df, label)["y"].values)

    # retrieve bodyaxis information per frame
    nose = label_array(df=df, label="nose")
    tail = label_array(df=df, label="T1")

    R, offset = R_per_frame(xnose_array=nose.x, ynose_array=nose.y,
                            xtail_array=tail.x, ytail_array=tail.y,
                            frames=frames_array(df))

    # y-axis offset
    middle_bodypoint = -.5*np.mean(tail.y)
    for x in np.arange(len(offset)):
        offset[x][1] += middle_bodypoint

    """
    -------------------------------------
    plotting config
    -------------------------------------
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # specific color assignment per label
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

    # axes initialisation
    fig_scatter, axes_scatter = plt.subplots(1, 2, sharex=True, sharey=True)

    fig_ellipse = plt.figure(tight_layout=True)
    grid_ellipse = gridspec.GridSpec(2, 3)
    axes_ellipse = [fig_ellipse.add_subplot(grid_ellipse[:, 0]),
                    fig_ellipse.add_subplot(grid_ellipse[0, 1]),
                    fig_ellipse.add_subplot(grid_ellipse[0, 2]),
                    fig_ellipse.add_subplot(grid_ellipse[1, 1:])]

    axes_ellipse[1].set_title(r"$\sigma$, b"), axes_ellipse[2].set_title(r"$\sigma$, a")
    axes_ellipse[3].set_title(r"$\theta$")

    axes_ellipse[1].set_xticks(np.arange(8)), axes_ellipse[2].set_xticks(np.arange(8))

    ellipsebodyparts = ['RF', 'LF', 'RB', 'LB', 'T2', 'T3', 'T4', 'T5']
    axes_ellipse[1].set_xticklabels(ellipsebodyparts, rotation=65), axes_ellipse[2].set_xticklabels(ellipsebodyparts,
                                                                                                    rotation=65)
    axes_ellipse[3].invert_yaxis()
    axes_ellipse[3].set_yticks(np.arange(8)), axes_ellipse[3].set_yticklabels(ellipsebodyparts)

    axes_ellipse[3].set_xticks(np.arange(-90, 90, 10)), axes_ellipse[3].set_xticklabels(
        [str(degree) + u'\xb0' for degree in np.arange(-90, 90, 10)])

    axes_ellipse[1].set_xlabel("px"), axes_ellipse[2].set_xlabel("px")

    """
    -------------------------------------
    Generating the plots
    -------------------------------------
    """
    import math
    from skimage.measure import EllipseModel # ellipse fitting
    from matplotlib.patches import Ellipse # ellipse plotting
    from scipy.stats import gaussian_kde # density plot

    # labels of interest
    bodyparts = [label for label in bodyparts_list(df) if "side" not in label]
    bodyparts.remove("nose")
    tails = [label for label in bodyparts if "T" in label]
    tails.remove("T1")

    print(bodyparts, "to be plotted features")
    print(tails, "to be plotted features tail parts")

    # initialize parameters
    ellipse_parameters = {}
    t = 0
    for idx, bodypart in enumerate(bodyparts):
        """
        data to be plotted
        """
        data = relative_to_body_array(bodypart_array=label_array(df=df, label=bodypart),
                                      R=R, offset=offset)

        """
        scatterplot
        """
        axes_scatter[0].scatter(data.x, data.y,
                                label=bodypart, color=colorlist[bodypart], alpha=.6)
        """
        ellipse fitting plot
        """
        if bodypart != "T1":
            axes_ellipse[0].scatter(data.x, data.y,
                                    label=bodypart, color=colorlist[bodypart], alpha=.6)

            # ellipse fitting
            ell = EllipseModel()
            ell.estimate(np.stack([data.x, data.y], axis=-1))

            xc, yc, a, b, theta = ell.params
            ellipse_parameters.update(
                {bodypart: {"xcenter": xc, "ycenter": yc, "a": a, "b": b, "theta": math.degrees(theta)}})

            # bar plots of ellipse information
            bar0 = axes_ellipse[1].bar([idx - t], b, color=colorlist[bodypart])
            axes_ellipse[2].bar([idx - t], a, color=colorlist[bodypart])
            axes_ellipse[3].barh([idx - t], math.degrees(theta), color=colorlist[bodypart])

            # mark center of the cluster
            axes_ellipse[0].scatter(xc, yc, s=25, color="maroon", marker="d")
            std_color = {1: "silver", 2: "gray"}
            for nstd in [1, 2]:
                ell_patch = Ellipse((xc, yc),
                                    2 * nstd * a,
                                    2 * nstd * b,
                                    theta * 180 / np.pi,
                                    edgecolor=std_color[nstd], facecolor="none", linestyle="--",
                                    label=str(nstd) + "sigma")
                axes_ellipse[0].add_patch(ell_patch) # plot fitted ellipse
        else:
            t = 1

        # gather density of cluster in "z" information
        xy = np.vstack([data.x, data.y])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()

        xnew, ynew, znew = data.x[idx], data.y[idx], z[idx]
        # plot density of cluster and cax, for colorbar
        cax = axes_scatter[1].scatter(x=xnew, y=ynew, c=znew, s=30,
                                      edgecolor='',
                                      cmap=plt.cm.Spectral_r,
                                      alpha=0.5)

    # colorbar of density plot
    cbar = fig_scatter.colorbar(cax)
    cbar.ax.get_yaxis().set_ticks([])
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel("High Density to Low Density", rotation=270)
    # nose marker
    axes_scatter[0].scatter(x=0, y=0 + np.abs(middle_bodypoint), color=colorlist["nose"], marker="X", label="nose")
    axes_ellipse[0].scatter(x=0, y=0 + np.abs(middle_bodypoint), color=colorlist["nose"], marker="X", label="nose")
    axes_ellipse[0].scatter(x=0, y=0 - np.abs(middle_bodypoint)/2, color=colorlist["T1"], marker="X",
                            label="T1")
    axes_scatter[1].scatter(x=0, y=0 + np.abs(middle_bodypoint), color=colorlist["nose"], marker="X")
    # legend
    axes_scatter[0].legend()
    axes_scatter[1].legend()
    # plot tail parts
    x_tailcoordinates, y_tailcoordinates = [0], [-np.abs(middle_bodypoint)/2]
    for T in tails:
        x_tailcoordinates.append(ellipse_parameters[T]["xcenter"])
        y_tailcoordinates.append(ellipse_parameters[T]["ycenter"])
    axes_ellipse[0].plot(x_tailcoordinates, y_tailcoordinates, color="maroon", alpha=.3)

    plt.show()