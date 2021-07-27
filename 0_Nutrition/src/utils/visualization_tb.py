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

##################################################### HEALTH DATA FUNCTIONS #####################################################
####
class eda_plotter():
    #####
    @staticmethod
    def __n_rows(df, n_columns):
        '''
        It calculates the number of rows (for the axes) depending on the number of variables to plot and the columns we want for the figure.
        args:
        n_columns: number of columns
        '''
        columns = list(df.columns)

        if len(columns) % n_columns == 0:
            axes_rows = len(columns) // n_columns
        else:
            axes_rows = (len(columns) // n_columns) + 1

        return axes_rows

    #####
    @staticmethod
    def rows_plotter(df, features_names, n_columns, kind = "box", figsize = (12, 6)):
        '''
        It plots all the variables in one row. It returns a figure
        args:
        n_columns: number of columns for the row
        kind: ("strip", "dist", "box")
        figsize: size of the figure
        '''
        # creates a figure with one axis and n_columns
        fig, axes = plt.subplots(1, n_columns, figsize = figsize)
        count = 0

        # Loop thorugh the generated axes
        for column in range(n_columns):
            if kind == "strip":
                sns.stripplot(y = df.iloc[:, count], ax = axes[column])
            elif kind == "dist":
                sns.distplot(df.iloc[:, count], ax = axes[column])
            elif kind == "box":
                sns.boxplot(df.iloc[:, count], ax = axes[column])
            else:
                sns.histplot(df.iloc[:, count], ax = axes[column], bins = 30)

            try:
                axes[column].set(xlabel = features_names[count])
            except:
                pass

            if (count + 1) < df.shape[1]:
                    count += 1
            else:
                break

        return fig

    #####
    @staticmethod
    def multi_axes_plotter(df, features_names, n_columns, kind = "box", figsize = (12, 12)):
        '''
        It creates a plot with multiple rows and columns. It returns a figure.
        n_columns: number of columns for the row
        kind: ("strip", "dist", "box")
        figsize: size of the figure
        '''
        # Calculating the number of rows from number of columns and variables to plot
        n_rows_ = eda_plotter.__n_rows(df, n_columns)

        # Creating the figure and as many axes as needed
        fig, axes = plt.subplots(n_rows_, n_columns, figsize = figsize)
        # To keep the count of the plotted variables
        count = 0

        # Some transformation, because with only one row, the shape is: (2,)
        axes_col = axes.shape[0]
        try:
            axes_row = axes.shape[1]
        except:
            axes_row = 1

        # Loop through rows
        for row in range(axes_col):
            # Loop through columns
            for column in range(axes_row):
                if kind == "strip":
                    sns.stripplot(y = df.iloc[:, count], ax = axes[row][column])
                elif kind == "dist":
                    sns.distplot(df.iloc[:, count], ax = axes[row][column])
                elif kind == "box":
                    sns.boxplot(df.iloc[:, count], ax = axes[row][column])
                else:
                    sns.histplot(df.iloc[:, count], ax = axes[row][column], bins = 30)

                try:
                    axes[row][column].set(xlabel = features_names[count])
                except:
                    pass

                if (count + 1) < df.shape[1]:
                    count += 1
                else:
                    break
        return fig

    #####
    @staticmethod
    def correlation_matrix(df, features_names, figsize = (12, 12)):
        '''
        It plots a correlation matrix. It returns a figure
        '''
        fig = plt.figure(figsize = figsize)
        sns.heatmap(df.corr(), annot = True, linewidths = .1,
                    cmap = "Blues", xticklabels = False,
                    yticklabels = features_names, cbar = False)

        return fig

####
class ml_model_plotter():
    '''
    Class to plot machine learning model results
    '''
    #####
    @staticmethod
    def train_val_plot(ml_model, figsize = (14, 6)):
        '''
        It plots training scores vs validation scores. It returns a figure
        '''
        fig = plt.figure(figsize = figsize)
        sns.set_theme()

        sns.lineplot(data = [ml_model.train_scores, ml_model.val_scores], markers = True, dashes = False)

        plt.ylabel("Score")
        plt.xlabel("Round")
        plt.legend(["Train score", "Validation score"])
        
        return fig

    #####
    @staticmethod
    def test_metrics(ml_model, figsize = (12, 12)):
        '''
        It plots the metrics after training with the full training data and testing with the test data. It returns a figure
        '''
        # Calculate the row/column totals for later use
        row_sums = ml_model.cm.sum(axis = 1, keepdims = True)
        column_sums = ml_model.cm.sum(axis = 0, keepdims = True)
        
        # Relative values to column/row sums
        rel_row = (ml_model.cm / row_sums) * 100
        rel_col = (ml_model.cm / column_sums) * 100

        # Plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = figsize, sharex = True, sharey = True)

        first_row_palette = sns.color_palette("light:b", as_cmap=True)
        second_row_palette = sns.light_palette("seagreen", as_cmap=True)
        fmt = "g"

        # ax1
        sns.heatmap(ml_model.cm, annot = True, linewidths = .1, cmap = first_row_palette, ax = ax1, cbar = False, fmt = fmt)
        ax1.set_ylabel("Actual class")
        ax1.set_title("Confusion matrix")

        # ax2
        sns.heatmap((ml_model.cm / ml_model.cm.sum()) * 100, annot = True, linewidths = .1, cmap = first_row_palette, ax = ax2, cbar = False, fmt = fmt)
        ax2.set_ylabel("Actual class")
        ax2.set_title("Confusion matrix - relative")

        # ax3
        sns.heatmap(rel_row, annot = True, linewidths = .1, cmap = second_row_palette, ax = ax3, cbar = False, fmt = fmt)
        ax3.set_xlabel("Predicted class")
        ax3.set_title("Relative to row sum (Recall)")

        # ax4
        sns.heatmap(rel_col, annot = True, linewidths = .1, cmap = second_row_palette, ax = ax4, cbar = False, fmt = fmt)
        ax4.set_xlabel("Predicted class")
        ax4.set_title("Relative to col sum (Precision)")

        return fig