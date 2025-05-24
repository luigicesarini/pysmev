import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union

def SMEV_FIG_valid(
    AMS: pd.DataFrame,
    RP: list,
    smev_RL: Union[np.ndarray, list] = [],
    smev_RL_unc=0,
    obscol_shape="g+",
    smev_colshape="--r",
    obslabel="Observed annual maxima",
    smevlabel="The SMEV model",
    alpha=0.2,
    xlimits: list = [1, 200],
    ylimits: list = [0, 50],
) -> None:
    """Plots figure return levels with annual maxima
    Args:
        AMS (pd.DataFrame): Dataframe containing annual maxima.
        RP (list): Return periods to plot.
        smev_RL (Union[np.ndarray, list], optional): Return levels calculated by SMEV. Defaults to [].
        smev_RL_unc (int, optional): Uncertainty of return levels calculated by SMEV. Only relevant if `smev_RL` is provided. Defaults to 0.
        obscol_shape (str, optional): Linestyle for annual maxima data to use in plot. Defaults to "g+".
        smev_colshape (str, optional): Linestyle for SMEV data to use in plot. Defaults to "--r".
        obslabel (str, optional): Label for annual maxima observation data to use in plot. Defaults to "Observed annual maxima".
        smevlabel (str, optional): Label for SMEV data to use in plot. Defaults to "The SMEV model".
        alpha (float, optional): Transparency to use in plot. Defaults to 0.2.
        xlimits (list, optional): x limits of plot. Defaults to [1, 200].
        ylimits (list, optional): y limits of plot. Defaults to [0, 50].
    """
    AMS_sort = AMS.sort_values(by=["AMS"])["AMS"]
    plot_pos = np.arange(1, np.size(AMS_sort) + 1) / (1 + np.size(AMS_sort))
    eRP = 1 / (1 - plot_pos)
    if np.size(smev_RL) != 0:
        # calculate uncertainty bounds. between 5% and 95%
        smev_RL_up = np.quantile(smev_RL_unc, 0.95, axis=0)
        smev_RL_low = np.quantile(smev_RL_unc, 0.05, axis=0)
        # plot uncertainties
        plt.fill_between(
            RP, smev_RL_low, smev_RL_up, color=smev_colshape[-1], alpha=alpha
        )  # SMEV

    plt.plot(eRP, AMS_sort, obscol_shape, label=obslabel)  # plot observed return levels
    if np.size(smev_RL) != 0:
        plt.plot(RP, smev_RL, smev_colshape, label=smevlabel)  # plot SMEV return lvls

    plt.xscale("log")
    plt.xlabel("return period (years)")
    plt.xticks(RP, labels=RP)
    plt.legend()
    plt.xlim(xlimits[0], xlimits[1])
    plt.ylim(ylimits[0], ylimits[1])
    
