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
def full_comparison_plot(comparisons):
    '''
    This function plots the full comparison of foods: vs daily intake, carbs and fats, cholesterol, energy. It returns the figure, so you need to put a plt.show() after this function to avoid having it double plotted.

    args : comparisons -> list of dataframes. Usually the output of the full_comparison function.
    '''
    # Unpack the dataframes
    comparison_di, comparison_fats, comparison_cholesterol, comparison_energy = comparisons

    # Create a figure with 4 axes
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (20, 20))

    # list of food groups in the dataframes
    food_groups = comparison_di["%OfDI"].unique()
    # calculate the amount of colors we need for the plots
    n_colors = len(food_groups)

    # create the palette with the calculated amount of colors
    palette = sns.color_palette("Paired", n_colors = n_colors)

    # ax1 : Daily intake
    sns.barplot(x = "Values", y = "Nutrient", hue = "%OfDI", data = comparison_di, palette = palette, ax = ax1)

    ax1.axvline(x=100, color='r', linestyle='dashed')

    ax1.set_title("% Of the Recommended Daily Intake", fontdict = {'fontsize': 20,
        'fontweight' : "bold"}, pad = 15)

    # ax2: Fats
    # This one works for the three remaining axes
    sns.barplot(x = "Values", y = "Nutrient", hue = "Food group", data = comparison_fats, palette = palette, ax = ax2)

    ax2.set_title("Fats (g)", fontdict = {'fontsize': 20,
        'fontweight' : "bold"}, pad = 15)

    # ax3: Cholesterol
    sns.barplot(x = "Values", y = "Food group", data = comparison_cholesterol, palette = palette, ax = ax3)

    ax3.set_title("Cholesterol (mg)", fontdict = {'fontsize': 20,
        'fontweight' : "bold"}, pad = 15)

    # ax4: Energy
    sns.barplot(x = "Values", y = "Food group", data = comparison_energy, palette = palette, ax = ax4)

    ax4.set_title("Energy (Kcal)", fontdict = {'fontsize': 20,
        'fontweight' : "bold"}, pad = 15)

    fig.tight_layout(pad = 3)

    return fig