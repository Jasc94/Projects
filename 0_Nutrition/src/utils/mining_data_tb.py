import pandas as pd
import numpy as np

import re
from varname import nameof

import requests
from bs4 import BeautifulSoup
import html
import lxml

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

##################################################### GENERIC FUNCTIONS #####################################################
########### NUMBER PROCESSORS
####
def to_string(x):
    try:
        return str(x)
    except:
        return x

####
def to_float(x):
    try:
        return float(x)
    except:
        return x

####
def num_cleaning(x):
    try:
        return re.match(r'[\d]*[\.\d]*', x)[0]
    except:
        return x
    
####
def gram_to_liter(x):
    return x * 0.001

####
def liter_to_gram(x):
    return x * 1000

########### OTHER PROCESSORS
####
def mapper(data):
    try:
        data.shape[1]       # This is actually to check whether it is a DataFrame or not
        return data.applymap(num_cleaning).applymap(to_float)
    except:
        return data.map(num_cleaning).map(to_float)

##################################################### ENVIRONMENT DATA FUNCTIONS #####################################################
#################### Daily Intake ####################
class daily_intake:
    def __init__(self, gender, age):
        self.gender = gender.lower()
        self.age = age
        self.url = None
        self.data = None

    ####
    def __data_selection(self, df):
        '''
        The function goes to the daily intake csv file, where all the links are stored and with the given parameters, returns the corresponding url.

        args :
        gender -> male / female
        age -> multiple of 10, between 20 and 70
        df -> dataframe with the urls
        '''
        self.url = df[(df["gender"] == self.gender) & (df["age"] == self.age)]["url"].values[0]

    ####
    def __clean_data(self, s):
        # Clear number formats
        self.data = mapper(s)

        # Drop unnecessary column
        self.data = self.data.drop("Iodine")

        # Rename Series object
        self.data.name = "Daily Intake"

        # Rename index
        self.data.index = ["Protein (g)", "Water (g)", "Fiber, total dietary (g)", "Vitamin A, RAE (mcg)", "Thiamin (mg)", "Riboflavin (mg)", "Niacin (mg)", "Vitamin B-6 (mg)", "Vitamin B-12 (mcg)", "Folate, total (mcg)", "Vitamin C (mg)", "Calcium (mg)", "Iron (mg)", "Magnesium (mg)", "Potassium (mg)", "Sodium (mg)", "Zinc (mg)"]
        
        # Transform liter values to gram (for consistency purposes)
        self.data["Water (g)"] = liter_to_gram(self.data["Water (g)"])

    ####
    def get_data(self, df):
        '''
        This function takes the url (return by pick_daily_intake) and pulls the daily intake data from it. It returns a pandas Series

        args :
        url -> url where daily intake data is stored
        '''
        self.__data_selection(df)

        r = requests.get(self.url)
        soup = BeautifulSoup(r.text, "lxml")

        di_table = soup.find(id = "tbl-calc")
        di_rows = di_table.find_all("tr")

        di_dict = {}

        for row in di_rows:
            items = row.find_all("td")
            if len(items) > 1:
                di_dict[items[0].text] = items[1].text

        s = pd.Series(di_dict)
        self.__clean_data(s)

        return self.data

#################### Nutritional values ####################
####
class filter_tool:
    ####
    def rows_filter(df, filter_, positive = True):
        if positive:
            filtered_df = df[df["Category name"].isin(filter_)]
        else:
            filtered_df = df[~df["Category name"].isin(filter_)]
            
        return filtered_df

    ####
    def multiple_filter(df, filters_, positive = True):
        dfs = []
        if positive:
            for filter_ in filters_:
                filtered_df = rows_filter(df, filter_)
                dfs.append(filtered_df)

            final_df = pd.concat(dfs)

        else:
            final_df = df[~df["Category name"].isin(filters_)]

        return final_df


#################### Daily Intake & Nutritional values ####################
####
class comparator:
    def __init__(self, foods, daily_intake):
        self.foods = foods
        self.daily_intake = daily_intake
        self.comparison = self.__comparator()

    ####
    def __comparator(self):
        # Merge first foods series with daily intake series
        comparison = pd.merge(self.daily_intake, self.foods[0], how = "outer", left_index = True, right_index = True)

        # If there's more than one item in foods list...
        if len(self.foods) > 1:
            # then merge the rest of the items with the dataframe we just created
            for food in self.foods[1:]:
                comparison = pd.merge(comparison, food, how = "outer", left_index = True, right_index = True)


        # To conclude, iterate over all food elements
        for food in self.foods:
            # Calculate the % of the daily nutrient intake the food provides with
            comparison[f"Relative - {food.name}"] = (comparison.loc[:, food.name] / comparison.loc[:, "Daily Intake"]) * 100

        return comparison

    ####
    def to_plot(self):
        # We get the columns with the relative nutritional values of the foods
        rel_comparison = self.comparison.iloc[:, -len(self.foods):]

        # We'll save the dataframes in the following list
        relatives = []

        # Iterate over the columns in comparison
        for column in rel_comparison.columns:
            # Get the Series coresponding to the food column
            rel = rel_comparison.loc[:, column]
            # Get nutrients out of the index
            rel = rel.reset_index()
            # Add a column with the food name
            rel["Food"] = column[11:]
            # Rename the columns for later use
            rel.columns = ["Nutrient", "Comparison", "Food"]
            # add the dataframe to our list
            relatives.append(rel)

        # Once we have all the dataframes, we'll stack them together vertically and return it
        return pd.concat(relatives)