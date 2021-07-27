import streamlit as st

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from keras.models import load_model

import sys, os

# Helpers
abspath = os.path.abspath
dirname = os.path.dirname
sep = os.sep

# Update sys.path for in-house libraries
folder_ = dirname(abspath(__file__))
for i in range(1): folder_ = dirname(folder_)
sys.path.append(folder_)

# In-house libraries
import utils.folder_tb as fo
import utils.mining_data_tb as md
import utils.visualization_tb as vi

##################################################### LOADING DATA #####################################################
####
@st.cache
def get_data():
    # Path to data
    environment_data_path = fo.path_to_folder(2, "data" + sep + "environment")
    health_data_path = fo.path_to_folder(2, "data" + sep + "health")

    # Load data
    # Environment-related data
    resources_df = pd.read_csv(environment_data_path + "resources.csv", index_col = 0)
    nutrition_df = pd.read_csv(environment_data_path + "nutritional_values.csv", index_col = 0)
    daily_intake_df = pd.read_csv(environment_data_path + "daily_intakes.csv")

    # Health related data
    health_df = pd.read_csv(health_data_path + "7_cleaned_data" + sep + "cleaned_data.csv", index_col = 0)
    # Object for the variables' names
    vardata = md.variables_data()
    vardata.load_data(health_data_path + "6_variables" + sep + "0_final_variables.csv")
    
    return resources_df, nutrition_df, daily_intake_df, health_df, vardata

####
@st.cache
def get_models():
    # Models path
    models_path = fo.path_to_folder(2, "models")

    # Load models
    logistic = joblib.load(models_path + "BM_LogisticRegression.pkl")
    #nn = models.load_model(models_path + "NN.h5")
    
    return logistic

# Save dataframes and models as variables
resources_df, nutrition_df, daily_intake_df, health_df, vardata = get_data()
logistic = get_models()


##################################################### INTERFACE #####################################################
####
menu = st.sidebar.selectbox("Menu:",
                            options = ["Home", "Resources Facts", "Nutrition Facts", "Health Facts", "Glossary", "API"])

####
if menu == "Home":
    #da.home()
    st.title("Food, Environment & Health")


####
if menu == "Resources Facts":
    #da.resources_facts()
    # To choose between subsections
    submenu = st.sidebar.radio(label = "What do you want to do?", options = ["Resources facts", "Comparator"])

    #### Title
    st.title("This is the resources facts section")
    st.sidebar.subheader("Play around")

    #### User input
    chosen_resource = st.sidebar.selectbox('Choose a resource:', options = resources_df.columns[:-1])

    #### Subsection 1
    if submenu == "Resources facts":
        # User input
        entries = st.sidebar.slider(label = "Entries:", min_value = 10,
                                    max_value = 50, value = 10, step = 10)

        # Page title
        st.subheader(f"You are currently checking the top {entries} by **{chosen_resource}**")

        # Data filtering
        selection = resources_df[["Origin", chosen_resource]].sort_values(by = chosen_resource, ascending = False).head(entries)
        
        # Creating table/plots
        header = ["Food"] + list(selection.columns)
        data = selection.reset_index().T

        table = go.Figure(data = go.Table(
                            columnwidth = [40, 30, 30],
                            header = dict(values = header,
                                         fill_color = "#3D5475",
                                         align = "left",
                                         font = dict(size = 20, color = "white")),
                          cells = dict(values = data,
                                       fill_color = "#7FAEF5",
                                       align = "left",
                                       font = dict(size = 16),
                                       height = 30))
                          )

        mapper = {"Plant-based" : "blue", "Animal-based" : "red"}
        color_map = md.color_mapper(selection, "Origin", mapper)
        fig = px.bar(selection, x = selection.index, y = chosen_resource,
                     color = color_map.keys(), color_discrete_map = color_map)

        # Data visualization
        st.write(fig)
        st.write(table)
        #st.table(selection)

    #### Subsection 2
    else:        
        #### Filters
        measure = st.sidebar.radio("Measure", options = ["Median", "Mean"]).lower()

        #### Section title
        st.subheader(f"You are currently checking the {measure} for the resource **{chosen_resource}**")

        #### Data extraction and prep
        stats_object = md.stats

        stats = stats_object.calculate(resources_df, [chosen_resource])
        to_plot = stats_object.to_plot(stats)

        fig = px.bar(to_plot[to_plot["Mean_median"] == measure], x = "Resource", y = "Values",
                                color = "Origin", color_discrete_map = {"Animal-based" : "red", "Plant-based" : "blue"},
                                barmode = "group")

        #### Data visualization
        st.write(fig)
        st.table(stats.T)


####
if menu == "Nutrition Facts":
    #da.nutrition_facts()
    #### Section title
    st.title("This is the nutrition facts section")

    # To choose between subsections
    submenu = st.sidebar.radio(label = "What do you want to do?", options = ["Top products", "Food groups", "Foods"])

    # Common tools for this section
    filter_tool = md.filter_tool
    filtered_df = nutrition_df
    
    # Subsection 1
    if submenu == "Top products":
        # User input
        chosen_nutrient = st.sidebar.selectbox("Nutrient", options = nutrition_df.columns[3:-2])
        entries = st.sidebar.slider(label = "How many foods?", min_value = 5, max_value = 50, 
                                    value = 5, step = 5)

        #### Filters
        expander = st.beta_expander("You can filter the data using the checkboxes")

        # Expander to hide the filders
        with expander:
            # Two columns for the filters
            cols = st.beta_columns(2)
            # Positive filters on the left column
            cols[0].write("Positive filters")
            positive_filters = ["Milks", "Cheese", "Other Animal Products", "Meats", "Chicken", "Fish",
                                "Milk Substitutes", "Beans", "Soy Products", "Nuts", "Other Veggie Products"]

            # Empty list to save the marked checkboxes
            positive_checkboxes = []
            # Iterate over every checkbox
            for filter_ in positive_filters:
                checkbox = cols[0].checkbox(filter_)
                # If they are marked, then append it to the empty list
                if checkbox:
                    positive_checkboxes.append(filter_)

            # Negative filters on the right column
            cols[1].write("Negative filters")
            negative_filters = ["Others", "Baby Food", "Desserts And Snacks", "Drinks", "Sandwiches", "Prepared Dishes", "Sauces"]

            # Similar functioning as for positive filters
            negative_checkboxes = []
            for filter_ in negative_filters:
                checkbox = cols[1].checkbox(filter_)
                if checkbox:
                    negative_checkboxes.append(filter_)

        # If positive_checkboxes list isn't empty...
        if len(positive_checkboxes) > 0:
            # Then filter the data
            f_ = filter_tool.multiple_filter(positive_checkboxes)
            filtered_df = filter_tool.rows_selector(filtered_df, f_)

        # If negative_checkboxes list isn't empty...
        if len(negative_checkboxes) > 0:
            # Then filter the data further
            f_ = filter_tool.multiple_filter(negative_checkboxes)
            filtered_df = filter_tool.rows_selector(filtered_df, f_, False)
        
        # Then, we apply the column filter to the filtered dataframe
        table = filter_tool.column_selector(filtered_df, chosen_nutrient).head(entries)
        ### Data prep for visualization
        mapper = {"Plant-based" : "blue", "Animal-based" : "red", "Not Classified" : "grey"}
        color_map = md.color_mapper(table.set_index("Food name"), "Category 3", mapper)
        fig = px.bar(table, x = "Food name", y = chosen_nutrient,
                     color = color_map.keys(), color_discrete_map = color_map, height = 800)

        fig.update(layout_showlegend=False)

        st.write(fig)
        st.table(table)
    
    if submenu == "Food groups":
        #### User input
        st.sidebar.write("To calculate Daily Intake")
        gender = st.sidebar.radio("Gender", options = ["Male", "Female"]).lower()
        age = st.sidebar.slider(label = "Age", min_value = 20, max_value = 70, value = 20, step = 10)

        st.sidebar.write("To calculate nutrients")
        measure = st.sidebar.radio("Measure of center", options = ["Mean", "Median"]).lower()

        expander = st.beta_expander("Food group filters")
        with expander:
            col1, col2, col3 = st.beta_columns(3)

            food_groups_stats = md.nutrients_stats(nutrition_df, "Category 2", measure)
            food_groups = []

            # Checkboxes divided in three blocks (purely design reasons)
            for food_group in food_groups_stats.columns[:4]:
                checkbox = col1.checkbox(label = food_group)
                if checkbox:
                    food_groups.append(food_group)

            for food_group in food_groups_stats.columns[4:8]:
                checkbox = col2.checkbox(label = food_group)
                if checkbox:
                    food_groups.append(food_group)

            for food_group in food_groups_stats.columns[8:12]:
                checkbox = col3.checkbox(label = food_group)
                if checkbox:
                    food_groups.append(food_group)

        if len(food_groups) > 0:
            #### Data prep for visualization
            daily_intake_object = md.daily_intake(gender, age)
            daily_intake = daily_intake_object.get_data(daily_intake_df)

            # I get the series for each food group (as I need them that way later on) and save them in a list
            foods = [food_groups_stats[column] for column in food_groups_stats[food_groups].columns]

            # Make comparisons, using a comparator object
            comparator = md.comparator(foods, daily_intake)
            comparisons = comparator.get_comparisons()
            
            # Save the plot as a figure
            fig = vi.full_comparison_plot(comparisons)

            st.write(fig)
            st.table(comparator.daily_intake_table())
    
    if submenu == "Foods":
        #### User input
        st.sidebar.write("To calculate Daily Intake")
        gender = st.sidebar.radio("Gender", options = ["Male", "Female"]).lower()
        age = st.sidebar.slider(label = "Age", min_value = 20, max_value = 70, value = 20, step = 10)

        #### Data prep for visualization
        daily_intake_object = md.daily_intake(gender, age)
        daily_intake = daily_intake_object.get_data(daily_intake_df)

        st.subheader("Food filters")

        # Filter the data by food groups, to make easier for the user to find the foods he wants to compare
        food_group_filter = st.selectbox('Food groups:',
                                         options = nutrition_df["Category 2"].unique())

        # Button to show the foods in the chosen food group
        filter_button = st.button("Show foods")
        filtered_df = nutrition_df[nutrition_df["Category 2"] == food_group_filter]
        filtered_df = filtered_df["Food name"]

        #st.write(nutrition_df.head(2))

        # Form to enter the foods to extract the data for
        with st.form("Submit"):
            chosen_foods = st.text_area("Foods you want to compare. Make sure you enter one value each line")
            submit_button = st.form_submit_button("Submit")
        
        # Once the user sends the info, plot everything
        if submit_button:
            # As the input is sent as string, let's split it by the line break
            chosen_foods = chosen_foods.split("\n")
            chosen_foods = list(filter(None, chosen_foods))     # Filter the extra line breaks
            df_ = nutrition_df.set_index("Food name")
            to_plot = [df_.loc[food, :] for food in chosen_foods]

            # ### Data preparation
            comparator = md.comparator(to_plot, daily_intake)
            comparisons = comparator.get_comparisons()

            fig = vi.full_comparison_plot(comparisons)

            # ### Data visualization
            st.subheader(f"Visualization of\n{chosen_foods}")
            st.pyplot(fig)

        if filter_button:
            st.table(filtered_df)

####
if menu == "Health Facts":
    #da.health_facts()
    #### Title
    st.title("This is the health facts section")

    submenu = st.sidebar.radio(label = "What do you want to do?", options = ["Exploration", "Health Prediction"])
    
    if submenu == "Exploration":
        st.header("In this section, you can explore the relation between different health indicators: demographics, dietary, and more.")

        sort_by = st.sidebar.radio("Sort by:", options = ["Variable nomenclature", "Variable description"])

        translation = {
            "Variable nomenclature" : "vAr_nAmE",
            "Variable description" : "var_descr",
        }

        # Table
        table_header = ["Variable name", "Variable description"]

        to_show = vardata.df.iloc[:, [0, 1, -2]].sort_values(by = translation[sort_by])
        table_data = [to_show.iloc[:, 0].values,
                    to_show.iloc[:, 1].values
                    ]

        table = go.Figure(data = go.Table(
                        columnwidth = [40, 100],
                        header = dict(values = table_header,
                        fill_color = "#3D5475",
                        align = "left",
                        font = dict(size = 20, color = "white")),

                        cells = dict(values = table_data,
                        fill_color = "#7FAEF5",
                        align = "left",
                        font = dict(size = 16),
                        height = 30)
                        ))
        table.update_layout(height = 300, margin = dict(l = 0, r = 0, b = 0, t = 0))
        
        st.write(table)
    
        ##### SECTION 2: Chossing and plotting variables
        st.header("2) Choose and plot some variables")

        # Plot filters
        st.sidebar.subheader("2) Data plotting")
        y = st.sidebar.text_input("Choose your target variable (y):")
        X = st.sidebar.text_area("Choose your explanatory variables (X):")
        X = X.split("\n")

        button = st.sidebar.button("Submit selection")

    if submenu == "Health Prediction":
        pass

####
if menu == "Glossary":
    #da.glossary()
    pass

####
if menu == "API":
    #da.api()
    pass