import pandas as pd
import numpy as np

import re

import requests
from bs4 import BeautifulSoup
import html
import lxml

import json
import joblib

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from imblearn.over_sampling import SMOTE

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
    """It possible, the function transforms value to string. Else, it returns the same value

    Args:
        x (any): Any value can be entered to the function

    Returns:
        str: value converted to string
    """
    try:
        return str(x)
    except:
        return x

####
def to_float(x):
    """It possible, the function transforms value to float. Else, it returns the same value

    Args:
        x (any): Any value can be entered to the function

    Returns:
        float: value converted to float
    """
    try:
        return float(x)
    except:
        return x

####
def num_cleaning(x):
    """The function gets number out of strings if possible. For instance, "13.47 €" will be converted to 13.47.

    Args:
        x (str): numerical value in str format that needs cleaning

    Returns:
        float: value converted to float
    """
    try:
        return re.match(r'[\d]*[\.\d]*', x)[0]
    except:
        return x

####
def round_number(x, dec):
    """It possible, the function rounds a value. Else, it returns the same value

    Args:
        x (float): Any value can be entered to the function
        dec (int): Number of decimals to keep

    Returns:
        float: value round "dec" decimals
    """
    try:
        return round(x, dec)
    except:
        return x
    
####
def gram_to_liter(x):
    """Converts gram units to liter units

    Args:
        x (int, float): value to convert to liters

    Returns:
        float: x value converted to liters
    """
    return x * 0.001

####
def liter_to_gram(x):
    """Converts liter units to gram units

    Args:
        x (int, float): value to convert to grams

    Returns:
        float: x value converted to grams
    """
    return x * 1000

####
def mapper(data):
    """Applis num_cleaning and to_float to a dataframe

    Args:
        data (dataframe): Dataframe to apply "num_cleaning" and "to_float" functions on

    Returns:
        [type]: [description]
    """
    try:
        data.shape[1]       # This is actually to check whether it is a DataFrame or not
        return data.applymap(num_cleaning).applymap(to_float)
    except:
        return data.map(num_cleaning).map(to_float)

####
def read_json(fullpath):
    """It opens a json file

    Args:
        fullpath (str): Path to json file

    Returns:
        json: read json file
    """
    with open(fullpath, "r") as json_file:
        read_json_ = json.load(json_file)

    return read_json_

####
def read_json_to_dict(json_fullpath):
    """
    Read a json and return a object created from it.
    Args:
        json_fullpath: json fullpath

    Returns: json object.
    """
    try:
        with open(json_fullpath, 'r+') as outfile:
            read_json = json.load(outfile)
        return read_json
    except Exception as error:
        raise ValueError(error)

##################################################### ENVIRONMENT DATA FUNCTIONS #####################################################
#################### Resources ####################
class merger:
    """Toolkit class to merge columns
    """
    @staticmethod
    def __merge_cols(df, column1, column2):
        """This function combines two foods' values in the resources data. For instance, "Tofu" and "Tofu (soybeans)", as they are the same food, and one has the missing values of the other.

        Args:
            df (dataframe): Dataframe where columns to combine are
            column1 ([type]): It should be the name of the column 1 in the dataframe to merge
            column2 ([type]): It should be the name of the column 2 in the dataframe to merge

        Returns:
            dataframe: dataframe with the columns combined in one
        """

        # To store the new values of combining both columns
        new_values = []

        # Iterate through the length of the column1 (both columns should have the same length)
        for i in range(len(df.loc[column1])):
            # If column1 is nan, return the value of the other column
            if np.isnan(df.loc[column1][i]):
                new_values.append(df.loc[column2][i])
            # else, keep the one from column 1
            else:
                new_values.append(df.loc[column1][i])

        # Join the values together with an index (should be the same for both columns)
        # and transpose it
        df = pd.DataFrame(new_values, index = df.loc[column1].index, columns = [column1 + "_"])
        return df.T

    @staticmethod
    def multiple_merge_cols(df, cols_list):
        """This function allows you to combine multiple pairs of columns at the same time.

        Args:
            df (dataframe): Dataframe where columns are
            cols_list (list): List of lists, being every sublist the column pair to combine. Example: [["Tofu", "Tofu (soybeans)"], [...]]

        Returns:
            dataframe: Dataframe with all the column pairs combined
        """
        to_append = []
        for cols in cols_list:
            new = merger.__merge_cols(df, cols[0], cols[1])
            df = df.drop(cols, axis = 0)
            to_append.append(new)

        return df.append(to_append)

class resources_stats():
    """Toolkit class to get stats from resources dataframe
    """
    @staticmethod
    def table(df, columns, group_by):
        """It calculates the mean and median of a dataframe grouping by a given column.

        Args:
            df (dataframe): Dataframe with data of interest
            columns ([type]): Dataframe columns that we want to keep
            group_by ([type]): Column to group by in order to calculate the stats

        Returns:
            dataframe: Stats' dataframe
        """
        # Group by "group_by" and calculate the mean and median for every column in columns
        stats = df.groupby(group_by).agg({column : (np.mean, np.median) for column in columns})
        return stats

    @staticmethod
    def to_plot(stats, columns):
        """It returns a dataframe ready to be plotted.

        Args:
            stats (dataframe): Stats' dataframe
            columns ([type]): Dataframe columns that we want to filter by for later plotting

        Returns:
            dataframe: Ready-to-plot dataframe
        """
        # Filter the data to plot
        to_plot = stats.loc[:, columns]
        # Some extra processing
        to_plot = to_plot.unstack().reset_index()
        # Renaming the columns
        to_plot.columns = ["Resource", "Measure", "Origin", "Values"]

        return to_plot


#################### Daily Intake ####################
class daily_intake:
    
    @staticmethod
    def __data_selection(df, gender, age):
        """It returns the corresponding url based on age and gender.

        Args:
            df (dataframe): Dataframe where the urls are
            gender (str): female/male
            age (int): 10, 20, 30, 40, 50, 60, 70

        Returns:
            str: url to get daily intake data from
        """
        url = df[(df["gender"] == gender) & (df["age"] == age)]["url"].values[0]

        return url

    @staticmethod
    def __clean_data(s):
        """It prepares the data.

        Args:
            s (Series): Daily Intake Series returned from the scrapped url

        Returns:
            Series: Series with cleaned and ready-to-use data
        """
        # Clear number formats
        data = mapper(s)

        # Drop unnecessary column
        data = data.drop("Iodine")

        # Rename Series object
        data.name = "Daily Intake"

        # Rename index
        data.index = ["Protein (g)", "Water (g)", "Fiber, total dietary (g)", "Vitamin A, RAE (mcg)", "Thiamin (mg)", "Riboflavin (mg)", "Niacin (mg)", "Vitamin B-6 (mg)", "Vitamin B-12 (mcg)", "Folate, total (mcg)", "Vitamin C (mg)", "Calcium (mg)", "Iron (mg)", "Magnesium (mg)", "Potassium (mg)", "Sodium (mg)", "Zinc (mg)"]
        
        # Transform liter values to gram (for consistency purposes)
        data["Water (g)"] = liter_to_gram(data["Water (g)"])

        return data

    @staticmethod
    def get_data(df, gender, age):
        """It combines the two previous functions and returns ready-to-use data about recommended daily intake based on gender and age

        Args:
            df (dataframe): Dataframe with URLs
            gender (str): female/male
            age (int): 10, 20, 30, 40, 50, 60, 70

        Returns:
            Series: ready-to-use data about daily intake
        """
        # Get url from dataframe
        url = daily_intake.__data_selection(df, gender, age)

        # Make BeautifulSoup object out of the url
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "lxml")

        # Get the table of the BeautifulSoup object
        di_table = soup.find(id = "tbl-calc")
        # Get every row
        di_rows = di_table.find_all("tr")

        # Empty dict to store the info
        di_dict = {}

        # Iterate over every found row
        for row in di_rows:
            items = row.find_all("td")
            # If the list has more than one element (this means, field name and field description)
            if len(items) > 1:
                # Then, save it in our empty dict
                di_dict[items[0].text] = items[1].text

        # Make a Series out of the filled dict
        s = pd.Series(di_dict)
        # Clean it
        data = daily_intake.__clean_data(s)

        return data

#################### Nutritional values ####################
####
class filter_tool:
    """Filtering class
    """
    ####
    @staticmethod
    def food_filter(key):
        """Returns all subcategories within the chosen category.

        Args:
            key (str): Category to get data from

        Returns:
            list: list of subcategories included in the chosen category
        """
        # category = [subcategories]
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

        # Grouping categories in super-categories
        animal_products = milks + cheese + other_animal_products + meats + chicken + fish
        veggie_products = milk_substitutes + beans + soy_products + nuts + other_veggie_products

        # All categories together
        full = animal_products + veggie_products

        # Map to match the given key by the user with the corresponding list of subsections
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
                        "Veggie Products" : veggie_products,
                        "full" : full
                    }
        
        return filters_map[key]

    ####
    @staticmethod
    def multiple_filter(keys):
        """It returns a list of multiple filters.

        Args:
            keys (list): List with all the categories we want to get the data from

        Returns:
            list: List of lists. Example: [['Fish', 'Shellfish'], [...]]
        """
        final_list = []
        for key in keys:
            final_list = final_list + filter_tool.food_filter(key)

        return final_list

    ####
    @staticmethod
    def rows_selector(df, filter_, positive = True):
        """It returns a dataframe filtered by row

        Args:
            df (dataframe): dataframe to filter
            filter_ (list): List of categories to filter by
            positive (bool, optional): If True, the function filters by the given list. If False, it filters out using the given list. Defaults to True.

        Returns:
            dataframe: Filtered dataframe
        """
        if positive:
            filtered_df = df[df["Category name"].isin(filter_)]
        else:
            filtered_df = df[~df["Category name"].isin(filter_)]
            
        return filtered_df

    ####
    @staticmethod
    def rows_selectors(df, filters_, positive = True):
        """It returns a dataframe filtered by row. This method can apply multiple filters at once

        Args:
            df (dataframe): Dataframe to filter
            filters_ (list): List of list of categories to filter by
            positive (bool, optional): If True, the function filters by the given list. If False, it filters out using the given list. Defaults to True.

        Returns:
            dataframe: Filtered dataframe
        """
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
        """This function allows us to filter the columns of the dataframe by nutrient.

        Args:
            df (dataframe): Dataframe to filter
            nutrient (str): Nutrient to filter the dataframe by

        Returns:
            dataframe: Filtered dataframe
        """
        try:
            columns = ["Food name", "Category name", "Category 2", "Category 3", nutrient]
            return df[columns].sort_values(by = nutrient, ascending = False)
        except:
            return "More than one row selected"

    ####
    @staticmethod
    def create_category(df, new_category, initial_value, new_values):
        """It creates a new category for a given dataframe

        Args:
            df (dataframe): Dataframe to create a new category for
            new_category (str): Name for the new category
            initial_value (any): Initial value that will be used for fill in the new dataframe's column
            new_values (list): List with "new value" and "filter" to use when assigning the new value. Example: ["Milks", milks]

        Returns:
            [type]: [description]
        """
        # Create new column
        df[new_category] = initial_value

        for pair in new_values:
            # Get the index of the foods whose "Category name" appead in the list
            condition = df[df["Category name"].isin(pair[1])].index
            # Assign new value to those rows that match de condition
            df.loc[condition, new_category] = pair[0]

        return df

####
def nutrients_stats(df, category, measure = "mean", start = 3, end = -2):
    """It calculates the mean/median for a given dataframe and category.

    Args:
        df (dataframe): Dataframe to calculate the stats for
        category (str): Column name we want the stats to be calculated for
        measure (str, optional): If "mean", it will calculate the mean. If "median", it will calculate the median. Defaults to "mean".
        start (int, optional): Column to start at. Defaults to 3.
        end (int, optional): Column to finish at. Defaults to -2.

    Returns:
        dataframe: Stats dataframe
    """
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
    """Class to compare food and food groups
    """
    def __init__(self, foods, daily_intake):
        """Constructor

        Args:
            foods (list): foods or food groups to be compared
            daily_intake (Series): Information about the daily intake. It will be used to compared foods/food groups
        """
        # Initial given values
        self.foods = foods
        self.daily_intake = daily_intake

        # Calculated behind the scenes when object is created
        self.comparison_di = self.__daily_intake_comparator()
        self.comparison_fats = self.comparator(['Sugars, total (g)',
       'Carbohydrate (g)', 'Total Fat (g)', 'Fatty acids, total saturated (g)',
       'Fatty acids, total monounsaturated (g)',
       'Fatty acids, total polyunsaturated (g)'])
        self.comparison_chol = self.comparator(['Cholesterol (mg)'])
        self.comparison_kcal = self.comparator(["Energy (kcal)"])


    ####
    def daily_intake_table(self):
        """It puts together the info about the daily intake fullfilment for the different foods

        Returns:
            dataframe: Dataframe to show as table. It shows how good foods are with respect to the daily intake
        """
        # Merge first foods series with daily intake series
        comparison_di = pd.merge(self.daily_intake, self.foods[0], how = "left", left_index = True, right_index = True)

        # If there's more than one item in foods list...
        if len(self.foods) > 1:
            # then merge the rest of the items with the dataframe we just created
            for food in self.foods[1:]:
                comparison_di = pd.merge(comparison_di, food, how = "left", left_index = True, right_index = True)


        # To conclude, iterate over all food elements
        for food in self.foods:
            # Calculate the % of the daily nutrient intake the food provides with
            comparison_di[f"Relative - {food.name} (%)"] = (comparison_di.loc[:, food.name] / comparison_di.loc[:, "Daily Intake"]) * 100

        return comparison_di

    ####
    def __daily_intake_comparator(self):
        """It returns a ready-to-plot dataframe with information about daily intake fullfilment

        Returns:
            dataframe: Ready-to-plot dataframe with the comparison between the food(s) and the daily intakec
        """
        # We get the columns with the relative nutritional values of the foods
        rel_comparison = self.daily_intake_table().iloc[:, -len(self.foods):]

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
            rel.columns = ["Nutrient", "Value", "Food"]
            # add the dataframe to our list
            relatives.append(rel)

        # Once we have all the dataframes, we'll stack them together vertically and return it
        return pd.concat(relatives)

    ####
    def comparator(self, filter_):
        """It returns a ready-to-plot dataframe with information about the chosen nutrients

        Args:
            filter_ (list): List of nutrients to use for the analysis

        Returns:
            dataframe: Ready-to-plot dataframe with the information about the chosen nutrients
        """
        processed_foods = []

        for food in self.foods:
            # Filter food nutrients
            data = food[filter_]
            # Get nutrients' names out of the index
            data = data.reset_index()
            # We need a new Series object for the food name
            food_name = pd.Series([food.name for i in range(len(data))])
            # Concat everything together
            data = pd.concat([data, food_name], axis = 1)
            # Rename the columns
            data.columns = ["Nutrient", "Value", "Food"]
            # Append this new df to our list
            processed_foods.append(data)

        return pd.concat(processed_foods, axis = 0)

    ####
    def get_comparisons(self):
        return self.comparison_di, self.comparison_fats, self.comparison_chol, self.comparison_kcal


#################### Data Prep for Visualization ####################
#### 
def color_mapper(df, column, mapper):
    """It returns a dict matching value-color for a given dataframe column

    Args:
        df (dataframe): Dataframe where the info is
        column (str): Column name that we want to color-map
        mapper (dict): Dict with column values as keys and colors as values. Example: {"column value 1":"CSS Code"}

    Returns:
        dict: dict with pairs {column value:color}
    """
    # Empty dict
    color_map = {}

    # Iterate over every row of the given dataframe
    for ind, row in df.iterrows():
        # Iterate over every pair key:value of the given mapper
        for key, val in mapper.items():
            # For the given column in every row, if the value matches one of the keys in the mapper...
            if row[column] == key:
                # Add the pair to our empty dict
                color_map[ind] = val
    
    return color_map


##################################################### HEALTH DATA FUNCTIONS #####################################################
################# VARIABLE NAMES #################
class variables_data:
    """Class that contains the dataframe with all the information about variables as well as some useful methods
    """
    def __init__(self):
        """Constructor
        """
        self.df = None

    #########
    def load_data(self, data_path):
        """Method to load the data to the object

        Args:
            data_path (str): Path to variables' data
        """
        self.df = pd.read_csv(data_path, index_col = 0)
        #return self.df

    #########
    def var_descr_detector(self, var_name, cut = None, nom_included = False):
        """It returns the variable description for a given variable nomenclature

        Args:
            var_name (str): Nomenclature we want the description for
            cut ([type], optional): As some of the variable descriptions are too long, this parameters allows you to cut it by character amount. Defaults to None.
            nom_included (bool, optional): If True, the function will return the nomenclature + description as follows: nomenclature : description. Defaults to False.

        Returns:
            str: It returns the variable description for a given nomenclature
        """
        # Try to find a description for the given nomenclature
        try: 
            # If nomenclature included...    
            if nom_included:
                # Then, add the nomenclature and the description together
                descr = var_name + ": " + self.df[self.df["vAr_nAmE"] == var_name]["var_descr"].values[0][:cut]
            else:
                # Else, get just the description
                descr = self.df[self.df["vAr_nAmE"] == var_name]["var_descr"].values[0][:cut]
            return descr
        # If error, return the nomenclature
        except:
            return var_name

    #########
    def vars_descr_detector(self, var_names, cut = None, nom_included = False):
        """It returns the variable descriptions for several variables at once

        Args:
            var_names (list): Nomenclatures we want the description for
            cut ([type], optional): As some of the variable descriptions are too long, this parameters allows you to cut it by character amount. Defaults to None.
            nom_included (bool, optional): If True, the function will return the nomenclature + description as follows: nomenclature : description. Defaults to False.

        Returns:
            list: List of descriptions
        """
        var_names = [self.var_descr_detector(nom, cut, nom_included) for nom in var_names] 

        return var_names

    @staticmethod
    def final_variables():
        """It returns the description for a given variable. This only applies to cleaned dataframe

        Returns:
            str: variable description
        """
        f_variables = {"RIDAGEYR" : "Age",
                       "BPXDI1" : "Diastolic: Blood pressure (mm Hg)",
                       "BPXSY1" : "Systolic: Blood pressure (mm Hg)",
                       "BMXWT" : "Weight (kg)",
                       "BMXWAIST" : "Waist Circumference (cm)",
                       "LBXTC" : "Total Cholesterol (mg/dL)",
                       "LBXSGL" : "Glucose (mg/dL)",
                       "MEANCHOL" : "Cholesterol (gm)",
                       "MEANTFAT" : "Total Fat (g)",
                       "MEANSFAT" : "Total Saturated Fatty Acis (g)",
                       "MEANSUGR" : "Total Sugar (g)",
                       "MEANFIBE" : "Total Fiber (g)",
                       "MEANTVB6" : "Total Vitamin B6 (mg)"}
                       
        return f_variables

################# READING DATA FROM FILES #################
class dataset:
    """Class to process the health dataframes
    """
    def __init__(self):
        """Constructor
        """
        # Raw data
        self.__dfs_list = []
        self.__joined_dfs = {}
        self.__raw_df = None
        self.df = None

        # Processed data for ML
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.kfold = None

    ######### DATA PROCESSING #########
    #########
    def __read_data(self, data_path):
        """It reads all the files from a folder as dataframes, and saves them all in a dict with the name of the file as a key.

        Args:
            data_path (str): Path to data

        Returns:
            dict: Dict with pairs name:dataframe.
        """
        # Empty dict to save the dataframes
        data_dfs = {}

        # Iterate over all the files in the folder
        for file_ in os.listdir(data_path):
            # If different to history
            if file_ != "history":
                # Try to process it
                try:
                    # Path to file
                    filepath = data_path + sep + file_

                    # Reading as dataframe
                    df = pd.read_csv(filepath, index_col = 0)
                    df["SEQN"] = df["SEQN"].map(int)
                    df.set_index("SEQN", inplace = True)

                    # Saving it in a dictionary
                    dict_key = file_[:-4].lower()
                    data_dfs[dict_key] = df
                # If error, go to the next one
                except:
                    pass

        return data_dfs

    #########
    def __read_all_data(self, data_path, folders):
        """It reads all the files from several folders as dataframes, and saves them all in dicts with the name of the file as a key.

        Args:
            data_path (str): Path to data
            folders (list): List of strings with the folder names we want to process
        """
        # Iterate over all the folders
        for folder in folders:
            # Calculate the path to the specific folder
            folder_path = data_path + folder
            # Get the dict name:dataframe
            self.__dfs_list.append(self.__read_data(folder_path))

    #########
    def __concatenate_dfs(self, data_dfs):
        """It receives a dict of dataframes and combines them by name

        Args:
            data_dfs (dict): Dict with filename as key and dataframe as value

        Returns:
            dict: Dict of combined dataframes
        """
        # Empty dict to save the combined dfs
        files = {}
        count = 0

        # Iterate over all the elements in the given dict
        for key, dfs in data_dfs.items():
            # Remove the last 2 characters of the name
            # Example:
            # DEMO_H -> DEMO
            # DEMO_I -> DEMO
            # DEMO_J -> DEMO
            key_ = key[:-2]

            # For the first iteration...
            if count == 0:
                # Add the first key:value pair to our empty dict
                files[key_] = dfs
            # After that....
            else:
                # If the new key:value pair isn't in the dict...
                if key_ not in files.keys():
                    # Add a new one
                    files[key_] = dfs
                # If the value already exists...
                # Example:
                # DEMO_H -> DEMO (first iteration)
                # DEMO_I -> DEMO (second iteration)
                else:
                    # Then combine it with the existing dataframe
                    files[key_] = pd.concat([files[key_], dfs])

            # Keep counting
            count +=1

        return files

    #########
    def __concatenate_all_dfs(self):
        """It's built on top of __concatenate_dfs to process multiple dicts of dataframes at once
        """
        # Iterate over all the dicts
        for data_dfs in self.__dfs_list:
            # Save as variable the created dict of combined dfs (example: DEMO from DEMO_H, DEMO_I, DEMO_J)
            files = self.__concatenate_dfs(data_dfs)
            # Joined all the dicts in one macro dict and save it as object attribute
            self.__joined_dfs = {**self.__joined_dfs, **files}


    #########
    def __merge_dfs(self):
        """It joins all dataframes in the full dict into one macro dataframe
        """
        # Get the dict keys
        keys = list(self.__joined_dfs.keys())

        # It saves all dataframes as object attribute
        self.df = self.__joined_dfs.pop(keys[0])

        # Iterate over items of dict with all dataframes
        for name, df in self.__joined_dfs.items():
            # It merges the existing df with every new dataframe pulled from the dict of dataframes and saves it as the object attribute
            self.df = pd.merge(self.df, df, how = "outer", on = "SEQN")
            
    #########
    def __clean_rows(self):
        """It removes values (rows) of no interest for specific columns. Values such as 7 or 9 that represent either "No answer" or "No info"
        """
        important_values = [7.0, 9.0]
        # Asthma
        self.df = self.df[~self.df.MCQ010.isin(important_values)]
        # Heart problems
        self.df = self.df[~self.df.MCQ160B.isin(important_values)]
        self.df = self.df[~self.df.MCQ160C.isin(important_values)]
        self.df = self.df[~self.df.MCQ160D.isin(important_values)]
        self.df = self.df[~self.df.MCQ160E.isin(important_values)]
        self.df = self.df[~self.df.MCQ160F.isin(important_values)]

    def __update_target_values(self):
        """It replaces the 2s with 0s for potential target variables, for better handling of categorical variables
        """
        self.df.MCQ010 = self.df.MCQ010.replace(2, 0)
        self.df.MCQ160B = self.df.MCQ160B.replace(2, 0)
        self.df.MCQ160C = self.df.MCQ160C.replace(2, 0)
        self.df.MCQ160D = self.df.MCQ160D.replace(2, 0)
        self.df.MCQ160E = self.df.MCQ160E.replace(2, 0)
        self.df.MCQ160F = self.df.MCQ160F.replace(2, 0)

    #########
    def __clean_columns(self, correction_map):
        """It removes duplicated columns.

        Args:
            correction_map (dict): Dict whose keys are the columns to rename and the values are the new names for those columns
        """
        to_drop = [key[:-2] + "_y" for key in correction_map.keys()]
        self.df = self.df.drop(to_drop, axis = 1)
        self.df = self.df.rename(columns = correction_map)

    #########
    def __heart_disease(self):
        """It creates a new column using all coronary-related diseases (variables) as source. The objective is to have a new column where we can see if the participant has any kind of heart disease.
        """
        # We create the column and fill it in with NaN values, as the initial status (with no information) is that we don't know whether someone has o doesn't have a coronary disease
        self.df["MCQ160H"] = np.nan

        # Conditions to filter by any heart disease
        pos_cond_b = self.df.MCQ160B == 1
        pos_cond_c = self.df.MCQ160C == 1
        pos_cond_d = self.df.MCQ160D == 1
        pos_cond_e = self.df.MCQ160E == 1
        pos_cond_f = self.df.MCQ160F == 1

        # For those participants we do have the info for and we know they don't have any coronary disease
        neg_cond_b = self.df.MCQ160B == 0
        neg_cond_c = self.df.MCQ160C == 0
        neg_cond_d = self.df.MCQ160D == 0
        neg_cond_e = self.df.MCQ160E == 0
        neg_cond_f = self.df.MCQ160F == 0

        # Given the positive conditions, place a "1" in the column if they are matched
        self.df.loc[(pos_cond_b) | (pos_cond_c) | (pos_cond_d) | (pos_cond_e) | (pos_cond_f), "MCQ160H"] = 1
        # Given the negative conditions, place a "0" in the column if they are matched
        self.df.loc[(neg_cond_b) & (neg_cond_c) & (neg_cond_d) & (neg_cond_e) & (neg_cond_f), "MCQ160H"] = 0

    #########
    def load_data(self, data_path, folders, correction_map):
        """It loads the health data into the object

        Args:
            data_path (str): Path to data
            folders (list): List of strings, being the strings the folder names
            correction_map (dict): Dict whose keys are the columns to rename and the values are the new names for those columns
        """
        self.__read_all_data(data_path, folders)
        self.__concatenate_all_dfs()
        self.__merge_dfs()
        self.__clean_rows()
        self.__update_target_values()
        self.__clean_columns(correction_map)
        self.__heart_disease()
        # Dataset backup
        self.__raw_df = self.df
    
    ######### SUPPORT FUNCTIONS #########
    #########
    def filter_columns(self, features, inplace = False):
        """It filters the dataframe.

        Args:
            features (list): List of columns we want to filter by
            inplace (bool, optional): If True, it will modify the dataframe within the object. Defaults to False.

        Returns:
            [type]: [description]
        """
        # inplace == True
        if inplace:
            # Replace the existing dataframe within the object by the new one filtered using the given features
            self.df = self.df.loc[:, features]
        else:
            # Just return the filtered dataframe by the given features
            return self.df.loc[:, features]

    #########
    def drop_columns(self, columns):
        """It drops columns from the object dataframe

        Args:
            columns (list): List of strings with the column names to drop
        """
        self.df = self.df.drop(columns, axis = 1)

    #########
    def drop_nans(self):
        """It drop nans
        """
        self.df = self.df.dropna()

    #########
    def dummies_transform(self, variable, mapper):
        """It transforms categorical variables into dummies.

        Args:
            variable ([type]): Target column to be transformed
            mapper ([type]): To preprocess the values before transforming the column into dummies. The pair should be old_value:new_value. For instance, {0: "male", 1: "female}
        """
        # Mapping values
        self.df.loc[:, variable] = self.df.loc[:, variable].map(mapper)
        # Getting dummies
        self.df = pd.get_dummies(self.df, prefix = "", prefix_sep = "", columns = [variable])

    #########
    def __pair_mean(self, pair_list, new_name, drop_old = False):
        """It creates a new column by calculating the mean of two other.

        Args:
            pair_list (list): List of columns to calculate the mean of. Example: ["DR1TCHOL", "DR2TCHOL"]
            new_name (str): New column name
            drop_old (bool, optional): If True, it will remove the columns we used to calculated the mean of. Defaults to False.
        """
        # Create a new column using the given new_name
        # As values use the mean of the given columns
        self.df[new_name] = self.df.loc[:, pair_list].mean(axis = 1)
        
        # if drop_old == True
        if drop_old:
            # Then, replace the existing dataframe with the new one
            self.df = self.df.drop(pair_list, axis = 1)

    #########
    def pairs_mean(self, combination_list, drop_old = False):
        """It creates new columns in our dataframe by calculating the mean of a pair of existing columns

        Args:
            combination_list (list): List of lists, being every sublist the pair to calculate the name of and the new column name. Example: [["DR1TCHOL", "DR2TCHOL"], "DRTCHOL"] 
            drop_old (bool, optional): If True, it will remove the columns we used to calculated the mean of. Defaults to False.
        """
        # Iterate over all the combinations [[col1, col2], new_name]
        for combination in combination_list:
            # Create the new column based on the given ones
            self.__pair_mean(combination[0], combination[1], drop_old = drop_old)

    #########
    def reset_dataset(self):
        """In case we want to restore the dataset to its first status (when used load_data method)
        """
        self.df = self.__raw_df

    #########
    def model_data(self, split, cv, epochs = 1, scaler = False, balance = None, seed = 42): 
        """This function prepares the data to feed it into a Machine Learning model

        Args:
            split (float): Value between 0 and 1 to split the data into train and test.
            cv (int): Cross validation folds. It has to be a positive value
            epochs (int, optional): Cross validation epochs. It has to be a positive value. Defaults to 1.
            scaler (bool, optional): If True, it uses StandardScaler() to scale the data. Defaults to False.
            balance (float, optional): It allows to balance data for imbalanced datasets. The given value will be the new ratio between positive and negatives. Defaults to None.
            seed (int, optional): Seed pass into the random state of the balancer. Defaults to 42.
        """
        '''
        It allows us to prepare the data for Machine Learning training
        '''
        # Independent variables
        X = np.array(self.df.iloc[:, 1:])

        # Dependent variable
        y = np.array(self.df.iloc[:, 0])

        # Data scaling
        if scaler:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # Train-test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = split, random_state = seed)

        # Balancing data
        if balance != None:
            sm = SMOTE(sampling_strategy = balance, random_state = seed, n_jobs = -1)
            self.X_train, self.y_train = sm.fit_resample(self.X_train, self.y_train)

        # Cross validation
        self.kfold = RepeatedStratifiedKFold(n_splits = cv, n_repeats = epochs, random_state = seed)

    #########
    def full_model_data(self, scaler = False, balance = None, seed = 42):
        """This function returns the full data (no train-test split) to train the model once tested.

        Args:
            scaler (bool, optional): If True, it uses StandardScaler() to scale the data. Defaults to False.
            balance (float, optional): It allows to balance data for imbalanced datasets. The given value will be the new ratio between positive and negatives. Defaults to None.
            seed (int, optional): Seed pass into the random state of the balancer. Defaults to 42.

        Returns:
            tuple: Tuple of independent variables and target (X, y)
        """
        # Independent variables
        X = np.array(self.df.iloc[:, 1:])

        # Dependent variable
        y = np.array(self.df.iloc[:, 0])

        # Data scaling
        if scaler:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # Balancing data
        if balance != None:
            sm = SMOTE(sampling_strategy = balance, random_state = seed, n_jobs = -1)
            X, y = sm.fit_resample(X, y)

        return X, y