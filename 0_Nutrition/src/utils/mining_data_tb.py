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
    @staticmethod
    def food_filter(key):
        others = ['Formula, ready-to-feed', 'Formula, prepared from powder', 'Formula, prepared from concentrate', 'Sugar substitutes', 'Not included in a food category']
        baby_food = ['Baby food: yogurt', 'Baby food: snacks and sweets', 'Baby food: meat and dinners', ]
        desserts_and_snacks = ['Ice cream and frozen dairy desserts', 'Milk shakes and other dairy drinks', 'Cakes and pies', 'Candy not containing chocolate', 'Doughnuts, sweet rolls, pastries', 'Crackers, excludes saltines', 'Cookies and brownies', 'Biscuits, muffins, quick breads', 'Pancakes, waffles, French toast', 'Cereal bars', 'Nutrition bars', 'Saltine crackers', 'Pretzels/snack mix', 'Potato chips', 'Candy containing chocolate', 'Pancakes, waffles, French toast']
        drinks = ['Soft drinks', 'Diet soft drinks', 'Flavored or carbonated water', 'Other diet drinks', 'Beer', 'Liquor and cocktails', 'Wine', 'Nutritional beverages', 'Protein and nutritional powders', 'Sport and energy drinks', 'Diet sport and energy drinks']
        sandwiches = ['Burritos and tacos', 'Other sandwiches (single code)', 'Burgers (single code)', 'Egg/breakfast sandwiches (single code)', 'Frankfurter sandwiches (single code)', 'Frankfurter sandwiches (single code)', 'Vegetables on a sandwich']
        prepared_dishes = ['Rolls and buns', 'Egg rolls, dumplings, sushi', 'Pasta mixed dishes, excludes macaroni and cheese', 'Macaroni and cheese', 'Pizza', 'Meat mixed dishes', 'Stir-fry and soy-based sauce mixtures', 'Bean, pea, legume dishes', 'Seafood mixed dishes', 'Rice mixed dishes', 'Fried rice and lo/chow mein', 'Poultry mixed dishes']
        sauces = ['Dips, gravies, other sauces''Pasta sauces, tomato-based', 'Mustard and other condiments', 'Mayonnaise', 'Jams, syrups, toppings']
        
        milks = ['Lamb, goat, game', 'Human milk', 'Milk, reduced fat', 'Milk, whole', 'Milk, lowfat', 'Milk, nonfat', 'Flavored milk, whole', 'Yogurt, regular', 'Yogurt, Greek']
        cheese = ['Cheese', 'Cottage/ricotta cheese']
        other_animal_products = ['Eggs and omelets', 'Butter and animal fats']
        meats = ['Ground beef', 'Cold cuts and cured meats', 'Bacon', 'Pork', 'Liver and organ meats', 'Frankfurters', 'Sausages']
        chicken = ['Turkey, duck, other poultry', 'Chicken, whole pieces', 'Chicken patties, nuggets and tenders']
        fish = ['Fish', 'Shellfish']

        milk_substitutes = ['Milk substitutes']
        beans = ['Beans, peas, legumes']
        soy_products = ['Processed soy products']
        nuts = ['Nuts and seeds']
        other_veggie_products = ['Peanut butter and jelly sandwiches (single code)', 'Oatmeal']

        animal_products = milks + cheese + other_animal_products + meats + chicken + fish
        veggie_products = milk_substitutes + beans + soy_products + nuts + other_veggie_products


        filters_map = {
                        "Others" : others,
                        "Baby Food" : baby_food,
                        "Desserts And Snacks" : desserts_and_snacks,
                        "Drinks" : drinks,
                        "Sandwiches" : sandwiches,
                        "Prepared Dishes" : prepared_dishes,
                        "Sauces" : sauces,
                        "Milks" : milks,
                        "Cheese" : cheese,
                        "Other Animal Products" : other_animal_products,
                        "Meats" : meats,
                        "Chicken" : chicken,
                        "Fish" : fish,
                        "Milk Substitutes" : milk_substitutes,
                        "Beans" : beans,
                        "Soy Products" : soy_products,
                        "Nuts" : nuts,
                        "Other Veggie Products" : other_veggie_products,
                        "Animal Products" : animal_products,
                        "Veggie Products" : veggie_products
                    }
        
        return filters_map[key]

    ####
    @staticmethod
    def multiple_filter(keys):
        final_list = []
        for key in keys:
            final_list = final_list + filter_tool.food_filter(key)

        return final_list

    ####
    @staticmethod
    def rows_selector(df, filter_, positive = True):
        if positive:
            filtered_df = df[df["Category name"].isin(filter_)]
        else:
            filtered_df = df[~df["Category name"].isin(filter_)]
            
        return filtered_df

    ####
    @staticmethod
    def rows_selectors(df, filters_, positive = True):
        dfs = []
        if positive:
            for filter_ in filters_:
                filtered_df = filter_tool.rows_selector(df, [filter_])
                dfs.append(filtered_df)

            final_df = pd.concat(dfs)

        else:
            final_df = df[~df["Category name"].isin(filters_)]

        return final_df

    ####
    @staticmethod
    def column_selector(df, nutrient):
        '''
        This function allows us to filter the columns of the dataframe by nutrient.

        args:
        nutrientname : nutrient to filter on
        df : dataframe to apply the filter to
        '''
        try:
            columns = ["Food name", "Category name", "Category 2", "Category 3", nutrient]
            return df[columns].sort_values(by = nutrient, ascending = False)
        except:
            return "More than one row selected"

####
def nutrients_stats(df, category, measure = "mean", start = 3, end = -2):
    nutrients_list = list(df.iloc[:, start:end].columns)

    if measure == "mean":
        stats = df.groupby(category).agg({nutrient : np.mean for nutrient in nutrients_list})
    elif measure == "median":
        stats = df.groupby(category).agg({nutrient : np.median for nutrient in nutrients_list})
    else:
        return "Invalid input for measure"
    return stats.T

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
        comparison = pd.merge(self.daily_intake, self.foods[0], how = "left", left_index = True, right_index = True)

        # If there's more than one item in foods list...
        if len(self.foods) > 1:
            # then merge the rest of the items with the dataframe we just created
            for food in self.foods[1:]:
                comparison = pd.merge(comparison, food, how = "left", left_index = True, right_index = True)


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


#################### Resources ####################
class stats:
    #### Stats for a joint plot
    @staticmethod
    def calculate(df, resources_list):
        '''
        This function calculates the center measures for the given resources belonging to the given dataframe.

        args : 
        df -> dataframe with the resources (cleaned)
        resources_list -> resources to be compared
        '''
        stats = df.groupby("Origin").agg({resource : (np.mean, np.median) for resource in resources_list})
        
        return stats

    #### Transformation for easier visualization
    @staticmethod
    def to_plot(stats):
        '''
        This function organizes the dataframe in a way that can be then plot with a bar graph.

        args : stats -> dataframe with the stats for the resources
        '''
        to_plot = stats.unstack()
        to_plot = to_plot.reset_index()
        to_plot.columns = ["Resource", "Mean_median", "Origin", "Values"]
        return to_plot


#################### Data Prep for Visualization ####################
####
def color_mapper(df, column, mapper):
    color_map = {}

    for ind, row in df.iterrows():
        for key, val in mapper.items():
            if row[column] == key:
                color_map[ind] = val
    
    return color_map