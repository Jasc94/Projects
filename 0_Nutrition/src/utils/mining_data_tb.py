import pandas as pd
import numpy as np

import re
from varname import nameof

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
def round_number(x, dec):
    '''
    It tries to round a value. It it can't, it will return the value with no modification
    '''
    try:
        return round(x, dec)
    except:
        return x
    
####
def gram_to_liter(x):
    return x * 0.001

####
def liter_to_gram(x):
    return x * 1000

####
def mapper(data):
    try:
        data.shape[1]       # This is actually to check whether it is a DataFrame or not
        return data.applymap(num_cleaning).applymap(to_float)
    except:
        return data.map(num_cleaning).map(to_float)

####
def read_json(fullpath):
    '''
    This function reads the json an returns it in a format we can work with it

    args : fullpath -> path to the json to be read
    '''
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
class comparator:
    def __init__(self, foods, daily_intake):
        self.foods = foods
        self.daily_intake = daily_intake

        self.comparison_di = self.__daily_intake_comparator()
        self.comparison_fats = self.comparator(['Sugars, total (g)',
       'Carbohydrate (g)', 'Total Fat (g)', 'Fatty acids, total saturated (g)',
       'Fatty acids, total monounsaturated (g)',
       'Fatty acids, total polyunsaturated (g)'])
        self.comparison_chol = self.comparator(['Cholesterol (mg)'])
        self.comparison_kcal = self.comparator(["Energy (kcal)"])


    ####
    def daily_intake_table(self):
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
    color_map = {}

    for ind, row in df.iterrows():
        for key, val in mapper.items():
            if row[column] == key:
                color_map[ind] = val
    
    return color_map


##################################################### HEALTH DATA FUNCTIONS #####################################################
################# VARIABLE NAMES #################
class variables_data:
    '''
    Object that contains the dataframe with all the information about variables as well as some useful methods
    '''
    def __init__(self):
        self.df = None

    #########
    def load_data(self, data_path):
        self.df = pd.read_csv(data_path, index_col = 0)
        #return self.df

    #########
    def var_descr_detector(self, var_name, cut = None, nom_included = False):
        '''
        It receives the variable code and returns the description with the nomenclature included is necessary.
        args:
        var_name: variable code
        cut: to limit the string to X amount of characters
        nom_included: if set to True, it will return variable code + variable name
        '''
        try:     
            if nom_included:
                descr = var_name + ": " + self.df[self.df["vAr_nAmE"] == var_name]["var_descr"].values[0][:cut]
            else:
                descr = self.df[self.df["vAr_nAmE"] == var_name]["var_descr"].values[0][:cut]
            return descr
        except:
            return var_name

    #########
    def vars_descr_detector(self, var_names, cut = None, nom_included = False):
        '''
        It does the same as var_descr_detector but for multiple variables at the same time
        '''
        var_names = [self.var_descr_detector(nom, cut, nom_included) for nom in var_names] 

        return var_names

################# READING DATA FROM FILES #################
class dataset:
    '''
    Object that will hold information about dataframe as well as do some useful transformations and save a copy in case we need to go back to the unprocessed version of the dataframe
    '''
    def __init__(self):
        # Raw data
        self.__dfs_list = []
        self.__joined_dfs = {}
        self.__raw_df = None
        self.df = None

        # Processed data for ML
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.kfold = None

    ######### DATA PROCESSING #########
    #########
    def __read_data(self, data_path):
        '''
        It reads all the files from a folder as dataframes, and saves them all in a dict with the name of the file as a key.
        args:
        up_levels: steps to go up from current folder
        folder: where the files are located
        '''
        data_dfs = {}
        for file_ in os.listdir(data_path):
            if file_ != "history":
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
                except:
                    pass

        return data_dfs

    #########
    def __read_all_data(self, data_path, folders):
        '''
        It does the same as __read_data but for several folders at the same time
        args: same as __read_data
        '''
        for folder in folders:
            folder_path = data_path + folder
            self.__dfs_list.append(self.__read_data(folder_path))

    #########
    def __concatenate_dfs(self, data_dfs):
        '''
        It receives a dict of dataframes and combines them by name
        args:
        data_dfs: dict with filename as key and dataframe as value
        '''
        files = {}
        count = 0

        for key, dfs in data_dfs.items():
            key_ = key[:-2]

            if count == 0:
                files[key_] = dfs
            else:
                if key_ not in files.keys():
                    files[key_] = dfs
                else:
                    files[key_] = pd.concat([files[key_], dfs])

            count +=1

        return files

    #########
    def __concatenate_all_dfs(self):
        '''
        It does the same as __concatenate_dfs but for multiple dicts
        '''
        for data_dfs in self.__dfs_list:
            files = self.__concatenate_dfs(data_dfs)
            self.__joined_dfs = {**self.__joined_dfs, **files}


    #########
    def __merge_dfs(self):
        '''
        It combines all dfs processed into one
        '''
        keys = list(self.__joined_dfs.keys())
        self.df = self.__joined_dfs.pop(keys[0])

        for name, df in self.__joined_dfs.items():
            self.df = pd.merge(self.df, df, how = "outer", on = "SEQN")
            
    #########
    def __clean_rows(self):
        '''
        It removes values (rows) of no interest for specific columns. Values such as 7 or 9 that represent either "No answer" or "No info"
        '''
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
        '''
        It replaces the 2s with 0s for potential target variables
        '''
        self.df.MCQ010 = self.df.MCQ010.replace(2, 0)
        self.df.MCQ160B = self.df.MCQ160B.replace(2, 0)
        self.df.MCQ160C = self.df.MCQ160C.replace(2, 0)
        self.df.MCQ160D = self.df.MCQ160D.replace(2, 0)
        self.df.MCQ160E = self.df.MCQ160E.replace(2, 0)
        self.df.MCQ160F = self.df.MCQ160F.replace(2, 0)

    #########
    def __clean_columns(self, correction_map):
        '''
        It removes duplicated columns.
        args:
        correction_map: dict which keys are the columns to rename and the values are the new names for those columns
        '''
        to_drop = [key[:-2] + "_y" for key in correction_map.keys()]
        self.df = self.df.drop(to_drop, axis = 1)
        self.df = self.df.rename(columns = correction_map)

    #########
    def __heart_disease(self):
        '''
        It creates a new column using all cardiovascular-related ones as source. The objective is to have a new column where we can see if the participant has any kind of heart disease.
        '''
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
        '''
        It combines all previous steps to get clean and ready-to-use data
        '''
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
        '''
        It filters the dataframe.
        args:
        features: columns we want to filter by
        inplace: default = False. If True, it will modify the dataframe within the object.
        '''
        if inplace:
            self.df = self.df.loc[:, features]
        else:
            return self.df.loc[:, features]

    #########
    def drop_columns(self, columns):
        '''
        To drop columns
        '''
        self.df = self.df.drop(columns, axis = 1)

    #########
    def drop_nans(self):
        '''
        To drop nans
        '''
        self.df = self.df.dropna()

    #########
    def dummies_transform(self, variable, mapper):
        '''
        Transforms categorical variables into dummies.
        args:
        variable: target column to be transformed
        mapper: To preprocess the values before transforming the column into dummies.
        '''
        # Mapping values
        self.df.loc[:, variable] = self.df.loc[:, variable].map(mapper)
        # Getting dummies
        self.df = pd.get_dummies(self.df, prefix = "", prefix_sep = "", columns = [variable])
        #return df

    #########
    def __pair_mean(self, pair_list, new_name, drop_old = False):
        '''
        It creates a new column by calculating the mean of two other.
        args:
        pair_list: columns to calculate the mean of
        new_name: name for the new column
        drop_old: set to False by default. If True, it will remove the columns we used to calculated the mean of
        '''
        self.df[new_name] = self.df.loc[:, pair_list].mean(axis = 1)
        
        if drop_old:
            self.df = self.df.drop(pair_list, axis = 1)

    #########
    def pairs_mean(self, combination_list, drop_old = False):
        '''
        It does the same as __pair_mean but for several pairs at once.
        args:
        combination_list: [[var1, var2], new_var]
        drop_old: By default set to False. If True, it will remove the variables used to calculated the mean.
        '''
        for combination in combination_list:
            self.__pair_mean(combination[0], combination[1], drop_old = drop_old)

    #########
    def reset_dataset(self):
        '''
        In case we want to restore the dataset to its first status (when used load_data method)
        '''
        self.df = self.__raw_df

    #########
    def model_data(self, split, cv, epochs = 1, scaler = False, balance = None, seed = 42): 
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