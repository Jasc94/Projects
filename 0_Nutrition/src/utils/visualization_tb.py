import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import sys, os

# Helpers
abspath = os.path.abspath
dirname = os.path.dirname
sep = os.sep

# Update sys.path for in-house libraries
folder_ = dirname(abspath(__file__))
sys.path.append(folder_)

# In-house libraries
import folder_tb as fo
import mining_data_tb as md


##################################################### ENVIRONMENT DATA FUNCTIONS #####################################################
#################### Daily Intake & Nutritional values ####################
####
def full_comparison_plot(comparisons, fontsize = 18, legendsize = 20, figsize = (20, 20)):
    comparison_di, comparison_fats, comparison_chol, comparison_kcal = comparisons

    sns.set_theme()
    n_colors = len(comparison_kcal["Food"].unique())
    palette = sns.color_palette("Paired", n_colors = n_colors)

    fig, ax = plt.subplots(2, 2, figsize = (20, 20))

    # AX1
    sns.barplot(x = "Value", y = "Nutrient", hue = "Food", data = comparison_di, palette = palette, ax = ax[0][0])
    ax[0][0].axvline(x = 100, color = "r", linestyle = "dashed")

    ax[0][0].set_title("% Of the Recommended Daily Intake", fontdict = {'fontsize': 20, 'fontweight' : "bold"}, pad = 15)
    ax[0][0].tick_params(axis = 'y', which = 'major', labelsize = fontsize)
    ax[0][0].set_xlabel("")
    ax[0][0].set_ylabel("")
    ax[0][0].legend(prop={'size': legendsize})

    # AX2
    sns.barplot(x = "Value", y = "Nutrient", hue = "Food", data = comparison_fats, palette = palette, ax = ax[0][1])

    ax[0][1].set_title("Fats & Carbs (g)", fontdict = {'fontsize': 20, 'fontweight' : "bold"}, pad = 15)
    ax[0][1].tick_params(axis = 'y', which = 'major', labelsize = fontsize)
    ax[0][1].set_xlabel("")
    ax[0][1].set_ylabel("")
    ax[0][1].legend().set_visible(False)

    # AX3
    sns.barplot(x = "Value", y = "Nutrient", hue = "Food", data = comparison_chol, palette = palette, ax = ax[1][0])

    ax[1][0].set_title("Cholesterol (mg)", fontdict = {'fontsize': 20, 'fontweight' : "bold"}, pad = 15)
    ax[1][0].tick_params(axis = 'y', which = 'major', labelsize = fontsize)
    ax[1][0].set_xlabel("")
    ax[1][0].set_ylabel("")
    ax[1][0].legend().set_visible(False)

    # AX4
    sns.barplot(x = "Value", y = "Nutrient", hue = "Food", data = comparison_kcal, palette = palette, ax = ax[1][1])

    ax[1][1].set_title("Energy (kcal)", fontdict = {'fontsize': 20, 'fontweight' : "bold"}, pad = 15)
    ax[1][1].tick_params(axis = 'y', which = 'major', labelsize = fontsize)
    ax[1][1].set_xlabel("")
    ax[1][1].set_ylabel("")
    ax[1][1].legend().set_visible(False)

    fig.tight_layout(pad = 3)
    return fig