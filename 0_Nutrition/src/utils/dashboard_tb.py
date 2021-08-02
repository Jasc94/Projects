import streamlit as st

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

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
import utils.dashboard_tb as da


def main():
    ##################################################### GET DATA #####################################################
    ####
    @st.cache
    def get_documentation():
        # Path to data
        general_path = fo.path_to_folder(2, "")
        documentation_path = fo.path_to_folder(2, "documentation")

        # Data sources (for glossary)
        with open(documentation_path + "Data_sources.md", "r") as file_:
            sources_data = file_.read()

        # Project info for home
        project_info = md.read_json_to_dict(general_path + "info.json")

        # Structure - home
        project_structure_image = documentation_path + "project_structure.png"
        with open(documentation_path + "project_structure_explanation.md", "r") as file_:
            project_structure_info = file_.read()

        # About me
        with open(documentation_path + "about_me.md", "r") as file_:
            about_me = file_.read()

        return sources_data, project_info, project_structure_image, project_structure_info, about_me

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
        cleaned_health_df = pd.read_csv(health_data_path + "7_cleaned_data" + sep + "cleaned_data.csv", index_col = 0)
        raw_health_df = pd.read_csv(health_data_path + "7_cleaned_data" + sep + "raw_data.csv", index_col = 0)
        # Object for the variables' names
        vardata = md.variables_data()
        vardata.load_data(health_data_path + "6_variables" + sep + "0_final_variables.csv")

        return resources_df, nutrition_df, daily_intake_df, cleaned_health_df, raw_health_df, vardata

    ####
    @st.cache
    def get_models():
        # Models path
        models_path = fo.path_to_folder(2, "models")

        # Load models
        logistic = joblib.load(models_path + "BM_LogisticRegression.pkl")
        #nn = models.load_model(models_path + "NN.h5")

        # Load models' insights
        models1_insights = pd.read_csv(models_path + "model_comparison_noscale_nobalance.csv", index_col = 0)
        models2_insights = pd.read_csv(models_path + "model_comparison_scaled_balanced.csv", index_col = 0)
        
        return logistic, models1_insights, models2_insights

    ##################################################### LOAD DATA #####################################################
    # Project info and resources
    sources_data, project_info, project_structure_image, project_structure_info, about_me = get_documentation() 
    # Dataframes and objects
    resources_df, nutrition_df, daily_intake_df, cleaned_health_df, raw_health_df, vardata = get_data()
    # Machine learning models
    logistic, models1_insights, models2_insights = get_models()

    # Plotter for later use
    plotly_plotter = vi.plotly_plotter

    ##################################################### INTERFACE #####################################################
    ############################ Menu ############################
    menu = st.sidebar.selectbox("Menu:",
                                options = ["Home", "Resources Facts", "Nutrition Facts", "Health Facts", "Glossary", "API", "About me"])

    ############################ Home ############################
    if menu == "Home":
        resources_path = fo.path_to_folder(2, "resources")

        # Section 1: Title and project description
        st.title(project_info["project_name"])
        st.image(resources_path + "home.jpeg")
        st.subheader(project_info["project_title"])
        st.write(project_info["project_description"])

        # Section 2: Project structure
        expander = st.beta_expander("Information on project structure")
        with expander:
            st.image(project_structure_image)
            st.markdown(project_structure_info)
        


    ############################ Resources Facts ############################
    if menu == "Resources Facts":
        # To choose between subsections
        submenu = st.sidebar.radio(label = "Submenu:", options = ["Food & Resources", "Comparator"])

        #### Title
        st.title("This is the resources facts section")

        #### User input
        st.sidebar.subheader("Play around")
        chosen_resource = st.sidebar.selectbox('Choose a resource:', options = resources_df.columns[:-1])

        # Translator for plot labels -> It adds the units
        translator = {"Total emissions" : "Total emissions (kg)",
                        "Land use per 1000kcal" : "Land use in squared meters (m2)",
                        "Land use per kg" : "Land use in squared meters (m2)",
                        "Land use per 100g protein" : "Land use in squared meters (m2)",
                        "Freshwater withdrawls per 1000kcal" : "Freshwater withdrawls in liters (l)",
                        "Freshwater withdrawls per kg" : "Freshwater withdrawls in liters (l)",
                        "Freshwater withdrawls per 100g protein" : "Freshwater withdrawls in liters (l)"}

        

        ########### Subsection 1: Food & Resources ###########
        if submenu == "Food & Resources":
            #### User input
            entries = st.sidebar.slider(label = "Entries:", min_value = 10,
                                        max_value = 50, value = 10, step = 10)

            # Page title
            st.subheader(f"You are currently checking the top {entries} by **{chosen_resource}**")

            #### Data filtering
            selection = resources_df[["Origin", chosen_resource]].sort_values(by = chosen_resource, ascending = False).head(entries).applymap(lambda x: md.round_number(x, 2)).dropna()
            
            #### Creating table/plots
            # 1) Data table
            # 1.1) Choose header and data of the table
            header = ["Food"] + list(selection.columns)
            data = selection.reset_index().T

            # 1.2) Create table
            table = plotly_plotter.resources_table(data, header)

            # 2) Plot
            # 2.1) Some data we need for the plot
            # To color the bars
            mapper = {"Plant-based" : "blue", "Animal-based" : "red"}
            # Apply the mapper to the dataframe
            color_map = md.color_mapper(selection, "Origin", mapper)
            # labels for the axes using the translator
            labels_map = {chosen_resource : translator[chosen_resource], "index" : "Foods"}
            
            # 2.2) Create the plot with the chosen data
            fig = plotly_plotter.resources_comparator(data = selection, x = selection.index, y = chosen_resource, color = color_map.keys(), color_discrete_map = color_map, labels_map = labels_map, kind = "foods")

            #### Data visualization
            st.write(fig)       # Plot
            st.write(table)     # Table

        ########### Subsection 2: Comparator ###########
        if submenu == "Comparator":        
            #### Filters
            measure = st.sidebar.radio("Measure", options = ["Median", "Mean"]).lower()

            #### Subsection title
            st.subheader(f"You are currently checking the {measure} for the resource **{chosen_resource}**")

            #### Data extraction and prep
            # 1) Create stats object
            resources_stats = md.resources_stats()

            # 2) Calculate the stats that go into the table
            stats = resources_stats.table(resources_df, resources_df.columns[:-1], "Origin")

            # 3) Transformations before plotting
            # Calculate the stats
            to_plot = resources_stats.to_plot(stats, [chosen_resource])
            # Use the measure chosen by the user
            to_plot = to_plot[to_plot["Measure"] == measure]

            # 4) Plot
            # To color the bars
            color_map = {"Animal-based" : "red", "Plant-based" : "blue"}
            # labels for the axes using the translator
            labels_map = {"Values" : translator[chosen_resource]}

            # The actual plot
            fig = plotly_plotter.resources_comparator(data = to_plot, x = "Resource", y = "Values", color = "Origin", color_discrete_map = color_map, labels_map = labels_map, kind = "groups")
            
            #### Data visualization
            # Data plot
            st.write(fig)

            # Data table
            expander = st.beta_expander("Insights on the data")
            with expander:
                st.table(to_plot)


    ############################ Nutrition Facts ############################
    if menu == "Nutrition Facts":
        #### Section title
        st.title("This is the nutrition facts section")

        #### User input
        # To choose between subsections
        submenu = st.sidebar.radio(label = "Submenu:", options = ["Top products", "Food groups", "Foods"])
        st.sidebar.subheader("Play around")

        # Common tools for this section
        # Filtering object
        filter_tool = md.filter_tool
        # Create a copy of nutrition_df that we can manipulate
        filtered_df = nutrition_df.copy()
        
        ########### Subsection 1: Top products ###########
        if submenu == "Top products":
            #### User input
            # Nutrient
            chosen_nutrient = st.sidebar.selectbox("Nutrient", options = nutrition_df.columns[3:-2])
            # Top: 5, 10, 15, ... values
            entries = st.sidebar.slider(label = "How many foods?", min_value = 5, max_value = 50, 
                                        value = 5, step = 5)

            
            #### Filters
            expander = st.beta_expander("You can filter the data using the checkboxes")

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
                    # If they are marked
                    if checkbox:
                        # Then append it to the empty list
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

            #### Data filtering and processing
            # If positive_checkboxes list isn't empty...
            if len(positive_checkboxes) > 0:
                # Then filter the data
                # 1) Get all the subsections of the checkboxes
                f_ = filter_tool.multiple_filter(positive_checkboxes)
                # 2) Filter the data with the full list of items
                filtered_df = filter_tool.rows_selector(filtered_df, f_)

            # If negative_checkboxes list isn't empty...
            if len(negative_checkboxes) > 0:
                # Then filter the data further
                # 1) Get all the subsections of the checkboxes
                f_ = filter_tool.multiple_filter(negative_checkboxes)
                # 2) Filter out the full list of items
                filtered_df = filter_tool.rows_selector(filtered_df, f_, False)
            
            # Then, we apply the column filter to the filtered dataframe
            table = filter_tool.column_selector(filtered_df, chosen_nutrient).head(entries)

            #### Plotting
            # Extras for the plotter
            # Dict matching value:color
            mapper = {"Plant-based" : "blue", "Animal-based" : "red", "Not Classified" : "grey"}
            # To color the bars
            color_map = md.color_mapper(table.set_index("Food name"), "Category 3", mapper)

            # Creating the plotly figure
            fig = plotly_plotter.top_foods(data = table, x = "Food name", y = chosen_nutrient, color = color_map.keys(), color_discrete_map = color_map)

            #### Data visualization
            st.write(fig)
            st.table(table)
        
        ########### Subsection 2: Food groups ###########
        if submenu == "Food groups":
            #### User input
            st.sidebar.write("To calculate Daily Intake")
            gender = st.sidebar.radio("Gender", options = ["Male", "Female"]).lower()
            age = st.sidebar.slider(label = "Age", min_value = 20, max_value = 70, value = 20, step = 10)

            st.sidebar.write("To calculate nutrients")
            measure = st.sidebar.radio("Measure of center", options = ["Mean", "Median"]).lower()

            #### Filters
            expander = st.beta_expander("Food group filters")
            with expander:
                # 3 columns
                col1, col2, col3 = st.beta_columns(3)

                # Calculate food stats by food group
                food_groups_stats = md.nutrients_stats(nutrition_df, "Category 2", measure)
                # Empty list for the filters
                food_groups = []

                # Checkboxes divided in three blocks (purely design reasons)
                # Iterate over the first four columns
                for food_group in food_groups_stats.columns[:4]:
                    # checkbox per food_group
                    checkbox = col1.checkbox(label = food_group)
                    # If checkbox == True (marked)
                    if checkbox:
                        # Then append the food group to empty list
                        food_groups.append(food_group)

                # Same functioning as the first one
                for food_group in food_groups_stats.columns[4:8]:
                    checkbox = col2.checkbox(label = food_group)
                    if checkbox:
                        food_groups.append(food_group)

                # Same functioning as the first one
                for food_group in food_groups_stats.columns[8:12]:
                    checkbox = col3.checkbox(label = food_group)
                    if checkbox:
                        food_groups.append(food_group)

            # If length of the food groups list is bigger than 0, that means, user has checked any of the checkboxes
            if len(food_groups) > 0:
                #### Data prep for visualization
                di_object = md.daily_intake()
                daily_intake = di_object.get_data(daily_intake_df, gender, age)

                # I get the series for each food group (as I need them that way later on) and save them in a list
                foods = [food_groups_stats[column] for column in food_groups_stats[food_groups].columns]

                # Make comparisons, using a comparator object
                comparator = md.comparator(foods, daily_intake)
                # Save the comparisons as variable
                comparisons = comparator.get_comparisons()
                
                # Save the plot as a figure
                fig = vi.full_comparison_plot(comparisons)

                #### Data visualization
                st.write(fig)
                st.table(comparator.daily_intake_table())
        
        ########### Subsection 3: Foods ###########
        if submenu == "Foods":
            #### User input
            st.sidebar.write("To calculate Daily Intake")
            gender = st.sidebar.radio("Gender", options = ["Male", "Female"]).lower()
            age = st.sidebar.slider(label = "Age", min_value = 20, max_value = 70, value = 20, step = 10)

            #### Data prep for visualization
            di_object = md.daily_intake()
            daily_intake = di_object.get_data(daily_intake_df, gender, age)

            st.subheader("Food filters")

            # Filter the data by food groups, to make easier for the user to find the foods he wants to compare
            food_group_filter = st.selectbox('Food groups:',
                                            options = nutrition_df["Category 2"].unique())

            # Button to show the foods in the chosen food group
            filter_button = st.button("Show foods")
            filtered_df = nutrition_df[nutrition_df["Category 2"] == food_group_filter]
            filtered_df = filtered_df["Food name"]

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

                # Creating the figure
                fig = vi.full_comparison_plot(comparisons)

                #### Data visualization
                st.subheader(f"Visualization of\n{chosen_foods}")
                st.pyplot(fig)

            # This shows the foods in the food group chosen by the user
            if filter_button:
                st.table(filtered_df)

    ############################ Health Facts ############################
    if menu == "Health Facts":
        #### Title
        st.title("This is the health facts section")

        #### User input
        submenu = st.sidebar.radio(label = "Submenu:", options = ["Exploration", "Health Prediction", "ML Models"])
        st.sidebar.subheader("Play around")

        ########### Subsection 1 ###########
        if submenu == "Exploration":
            #### Subsubsection 1: initial interface
            # Subsection title
            st.subheader("In this section, you can explore the relation between different health indicators: demographics, dietary, and more.")

            # To sort the table
            sort_by = st.sidebar.radio("Sort by:", options = ["Variable nomenclature", "Variable description"])

            # To replace the names in the original dataframe so that they are easier to understand in the front
            translation = {
                "Variable nomenclature" : "vAr_nAmE",
                "Variable description" : "var_descr",
            }

            # Filters the full dataframe by category, so that users can see just demographics variables, dietary variables, etc...
            filter_by = st.sidebar.radio("Filter by:", options = ["Demographics", "Dietary", "Examination", "Laboratory", "Questionnaire"])

            # Table
            table_header = ["Variable name", "Variable description"]

            # It takes the columns of interest and sort the resulting dataframe by "sort_by"
            to_show = vardata.df.iloc[:, [0, 1, -2]].sort_values(by = translation[sort_by])
            # It filters the data by the chosen category
            to_show = to_show[to_show.component == filter_by]

            # It gets the relevant data
            table_data = [to_show.iloc[:, 0].values,
                        to_show.iloc[:, 1].values
                        ]

            # It creates the plotly table
            table = plotly_plotter.health_table(table_data, table_header)

            # data visualization            
            st.write(table)
        
            #### Subsubsection 2: Chossing and plotting variables
            st.subheader("Choose and plot some variables")

            # Plot filters
            st.sidebar.subheader("Data plotting")
            y = st.sidebar.text_input("Choose your target variable (y):")
            X = st.sidebar.text_area("Choose your explanatory variables (X):")
            X = X.split("\n")

            button = st.sidebar.button("Submit selection")

            #### Data insights
            if button:
                #### Data preprocessing
                features = [y] + X          # It joins target and independent variables
                data = raw_health_df.loc[:, features]       # It filters the data by "features"
                filtered_data = data.dropna()           # It removes the NaN values

                # Data description and some processing for later plotting
                data_stats = filtered_data.describe().T
                data_stats = data_stats.reset_index()
                data_stats = data_stats.drop("count", axis = 1)
                data_stats = data_stats.applymap(lambda x: md.round_number(x, 2))
                data_columns = list(data_stats.columns)

                # Calculate correlations
                corr = np.array(filtered_data.corr().applymap(lambda x: round(x, 2)))

                # Get variables' descriptions
                y_descr = vardata.var_descr_detector(y)
                X_descr = vardata.vars_descr_detector(X, cut = 30)
                descrs = [y] + X_descr

                # Stats table
                table_header = data_columns
                table_data = [data_stats.iloc[:, column].values for column in range(len(data_columns))]

                # Table with variables' info
                table = go.Figure(data = go.Table(
                                columnwidth = [20, 10, 10, 10, 10, 10, 10, 10],         # columns' width
                                # Header    
                                header = dict(values = table_header,
                                fill_color = "#3D5475",
                                align = "left",
                                font = dict(size = 20, color = "white")),
                                # Rest of the data
                                cells = dict(values = table_data,
                                fill_color = "#7FAEF5",
                                align = "left",
                                font = dict(size = 16),
                                height = 30)
                                ))

                # To adjust the height of the table and avoid as much as possible too much white space
                if len(features) < 6:
                    table.update_layout(autosize = False, width = 600, height = 150,
                                        margin = dict(l = 0, r = 0, b = 0, t = 0))
                else:
                    table.update_layout(autosize = False, width = 600, height = 200,
                                        margin = dict(l = 0, r = 0, b = 0, t = 0))

                #### Data insights
                # Expander
                expander = st.beta_expander("Insights on the data")
                with expander:
                    # Insights on the target variable
                    st.write("**Chosen variables**:")
                    for feature in features:
                        st.write(vardata.var_descr_detector(feature, nom_included = True))
                    
                    # General information
                    st.write("**Number of observations**:")
                    st.write(f"- Before dropping the NaN values:\t{data.shape[0]}")
                    st.write(f"- After dropping the NaN values:\t{filtered_data.shape[0]}")
                    st.write("\n")
                    st.write("**Target variable (y) values**:")
                    st.table(filtered_data.loc[:, y].value_counts())

                    # External link
                    nahnes_url = "https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017"
                    st.write("More info in the following link:")
                    st.markdown(nahnes_url, unsafe_allow_html=True)

                ### Correlation plot
                st.write(y_descr)
                colorscale = [[0, "white"], [1, "cornflowerblue"]]

                corr_plot = plotly_plotter.health_correlation(corr, descrs, colorscale)
                # Show the correlation plot
                st.write(corr_plot)

                # Distribution plots for each chosen variable
                for x in X:
                    x_descr = vardata.var_descr_detector(x, cut = 30, nom_included = True)
                    expander = st.beta_expander(x_descr)

                    # within expanders to ease the navigability
                    with expander: 
                        to_plot = filtered_data.loc[:, [y, x]].dropna()
                        labels = {x : x_descr}

                        hist = plotly_plotter.health_hist(to_plot, x, y, labels)
                        st.write(hist)

        ########### Subsection 2 ###########
        if submenu == "Health Prediction":
            st.sidebar.write("Predict whether or not you can have a coronary disease")
            predict_button = st.sidebar.button("Predict health")

            # Female average values
            female_info = cleaned_health_df[cleaned_health_df["Female"] == 1].describe()
            female_avg_val = female_info.loc["mean", "RIDAGEYR":].map(lambda x: round(x, 0))

            # Male average values
            male_info = cleaned_health_df[cleaned_health_df["Male"] == 1].describe()
            male_avg_val = male_info.loc["mean", "RIDAGEYR":].map(lambda x: round(x, 0))

            # Variable names
            fv = vardata.final_variables()

            # Form for the prediction
            expander = st.beta_expander("Find out if you are at risk of heart disease")

            with expander:
                cols = st.beta_columns(3)
                # Col 1
                GENDER = cols[0].selectbox("Gender", options = ["Female", "Male"])

                if GENDER == "Female":
                    FEMALE = 1
                    MALE = 0

                    # Rest of the user input
                    RIDAGEYR = cols[0].text_input(fv["RIDAGEYR"], value = female_avg_val["RIDAGEYR"])
                    BPXDI1 = cols[0].text_input(fv["BPXDI1"], value = female_avg_val["BPXDI1"])
                    BPXSY1 = cols[0].text_input(fv["BPXSY1"], value = female_avg_val["BPXSY1"])
                    BMXWT = cols[0].text_input(fv["BMXWT"], value = female_avg_val["BMXWT"])

                    # Col 2
                    BMXWAIST = cols[1].text_input(fv["BMXWAIST"], value = female_avg_val["BMXWAIST"])
                    LBXTC = cols[1].text_input(fv["LBXTC"] + "*", value = female_avg_val["LBXTC"])
                    LBXSGL = cols[1].text_input(fv["LBXSGL"] + "*", value = female_avg_val["LBXSGL"])
                    MEANCHOL = cols[1].text_input(fv["MEANCHOL"] + "**", value = female_avg_val["MEANCHOL"])
                    MEANTFAT = cols[1].text_input(fv["MEANTFAT"] + "**", value = female_avg_val["MEANTFAT"])

                    # Col 3
                    MEANSFAT = cols[2].text_input(fv["MEANSFAT"] + "**", value = female_avg_val["MEANSFAT"])
                    MEANSUGR = cols[2].text_input(fv["MEANSUGR"] + "**", value = female_avg_val["MEANSUGR"])
                    MEANFIBE = cols[2].text_input(fv["MEANFIBE"] + "**", value = female_avg_val["MEANFIBE"])
                    MEANTVB6 = cols[2].text_input(fv["MEANTVB6"] + "**", value = female_avg_val["MEANTVB6"])

                else:
                    FEMALE = 0
                    MALE = 1

                    # Rest of the user input
                    RIDAGEYR = cols[0].text_input(fv["RIDAGEYR"], value = male_avg_val["RIDAGEYR"])
                    BPXDI1 = cols[0].text_input(fv["BPXDI1"], value = male_avg_val["BPXDI1"])
                    BPXSY1 = cols[0].text_input(fv["BPXSY1"], value = male_avg_val["BPXSY1"])
                    BMXWT = cols[0].text_input(fv["BMXWT"], value = male_avg_val["BMXWT"])

                    # Col 2
                    BMXWAIST = cols[1].text_input(fv["BMXWAIST"], value = male_avg_val["BMXWAIST"])
                    LBXTC = cols[1].text_input(fv["LBXTC"] + "*", value = male_avg_val["LBXTC"])
                    LBXSGL = cols[1].text_input(fv["LBXSGL"] + "*", value = male_avg_val["LBXSGL"])
                    MEANCHOL = cols[1].text_input(fv["MEANCHOL"] + "**", value = male_avg_val["MEANCHOL"])
                    MEANTFAT = cols[1].text_input(fv["MEANTFAT"] + "**", value = male_avg_val["MEANTFAT"])

                    # Col 3
                    MEANSFAT = cols[2].text_input(fv["MEANSFAT"] + "**", value = male_avg_val["MEANSFAT"])
                    MEANSUGR = cols[2].text_input(fv["MEANSUGR"] + "**", value = male_avg_val["MEANSUGR"])
                    MEANFIBE = cols[2].text_input(fv["MEANFIBE"] + "**", value = male_avg_val["MEANFIBE"])
                    MEANTVB6 = cols[2].text_input(fv["MEANTVB6"] + "**", value = male_avg_val["MEANTVB6"])
                

                # Annotations
                st.write("\* Blood levels", value = 68)
                st.write("** Usual daily intake (diet habits)", value = 68)
                st.markdown("<i>Predictions are made using a Logistic Regresion Machine Learning model. If you want some more insights about the accuracy of this models, please head to the 'Models' section</i>", unsafe_allow_html = True)

                # Gathering all the form data
                to_predict = [RIDAGEYR, BPXDI1, BPXSY1, BMXWT, BMXWAIST, LBXTC,
                            LBXSGL, MEANCHOL, MEANTFAT, MEANSFAT, MEANSUGR, MEANFIBE,
                            MEANTVB6, FEMALE, MALE]

            # Predictions
            if predict_button:
                # Processing data for the model
                to_predict = [md.to_float(val) for val in to_predict]
                to_predict_arr = np.array(to_predict).reshape(1, -1)

                # Prediction
                ml_prediction = logistic.predict(to_predict_arr)
                
                # Note to the user
                st.sidebar.write("Scroll down to see the prediction!")

                #Showing the prediction
                if ml_prediction[0] == 1:
                    st.header("You are at risk of having a coronary disease")
                    st.write("This results are an estimation and you should always check with your doctor.")
                else:
                    st.header("You are not at risk of having a coronary disease")
                    st.write("This results are an estimation and you should always check with your doctor.")

                st.subheader("Your following values are above average:")
                st.markdown("<i>Average is calculated based on the data collected for this study</i>", unsafe_allow_html = True)

                count = 1
                if GENDER == "Female":
                    for ind, val in female_avg_val[1:-2].items():
                        if to_predict[count] > val:
                            analysis = f"{fv[ind]} | Average: {val} | Your value: {to_predict[count]}"
                            st.write(analysis)
                        count += 1

                if GENDER == "Male":
                    for ind, val in female_avg_val[1:-2].items():
                        if to_predict[count] > val:
                            analysis = f"{fv[ind]} | Average: {val} | Your value: {to_predict[count]}"
                            st.write(analysis)
                        count += 1
        
        ########### Subsection 3 ###########
        if submenu == "ML Models":
            st.subheader("Models without data scaling or balancing")
            st.write("These models were trained and tested using the raw data. This means, no scaling and no balancing of the data")
            st.table(models1_insights)

            st.subheader("Models with data scaling and balancing")
            st.write("These models were trained and tested using the modified data. This means, scaling and balancing the data")
            st.table(models2_insights)

            st.subheader("Conclusions")
            st.write("Although the models in the second table show lower scores, they reached higher recall levels, meaning that they were able to detect better positive cases of the coronary disease, which was the goal.\nFor this reason, the model used for the prediction section is the LogisticRegression with max_iter = 500 and warm start.")
            

    ############################ GLOSSARY ############################
    if menu == "Glossary":
        #da.glossary()
        st.write(sources_data)


    ############################ API ############################
    if menu == "API":
        #da.api()
        selection = st.sidebar.radio("Choose data:",
                                    options = ["Resources",
                                                "Nutrition",
                                                "Health",
                                                "Health variables"])
        url = "http://localhost:6060"

        if selection == "Resources":
            try:
                url = f"{url}/resources"
                data = pd.read_json(url)
                st.table(data.head(10))
            except:
                st.header("It wasn't possible to gather the data")
                st.write("Please confirm that the server is running")
        
        if selection == "Nutrition":
            try:
                url = f"{url}/nutrition"
                data = pd.read_json(url)
                st.table(data.head(10))
            except:
                st.header("It wasn't possible to gather the data")
                st.write("Please confirm that the server is running")

        if selection == "Health":
            try:
                url = f"{url}/health"
                data = pd.read_json(url)
                st.table(data.head(10))
            except:
                st.header("It wasn't possible to gather the data")
                st.write("Please confirm that the server is running")

        if selection == "Health variables":
            try:
                url = f"{url}/health-variables"
                data = pd.read_json(url)
                st.table(data.head(10))
            except:
                st.header("It wasn't possible to gather the data")
                st.write("Please confirm that the server is running")


    ############################ About me ############################
    if menu == "About me":
        st.markdown(about_me)