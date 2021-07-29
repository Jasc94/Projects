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

# Save dataframes and models as variables
resources_df, nutrition_df, daily_intake_df, cleaned_health_df, raw_health_df, vardata = get_data()
logistic, models1_insights, models2_insights = get_models()


##################################################### INTERFACE #####################################################
####
menu = st.sidebar.selectbox("Menu:",
                            options = ["Home", "Resources Facts", "Nutrition Facts", "Health Facts", "ML Models", "Glossary", "API"])

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
        selection = resources_df[["Origin", chosen_resource]].sort_values(by = chosen_resource, ascending = False).head(entries).applymap(lambda x: md.round_number(x, 2))
        
        # Creating table/plots
        header = ["Food"] + list(selection.columns)
        data = selection.reset_index().T

        table = go.Figure(data = go.Table(
                            columnwidth = [50, 50, 40],
                            header = dict(values = header,
                                         fill_color = "#5B5B5E",
                                         align = "left",
                                         font = dict(size = 20, color = "white")),
                          cells = dict(values = data,
                                       fill_color = "#CBCBD4",
                                       align = "left",
                                       font = dict(size = 16),
                                       height = 30))
                          )

        table.update_layout(height = 300, margin = dict(l = 0, r = 0, b = 0, t = 0))
        st.write(table)
        mapper = {"Plant-based" : "blue", "Animal-based" : "red"}
        color_map = md.color_mapper(selection, "Origin", mapper)
        fig = px.bar(selection, x = selection.index, y = chosen_resource,
                     color = color_map.keys(), color_discrete_map = color_map)

        fig.update(layout_showlegend=False)

        # Data visualization
        st.write(fig)
        st.write(table)
        #st.table(selection)

    #### Subsection 2
    if submenu == "Comparator":        
        #### Filters
        measure = st.sidebar.radio("Measure", options = ["Median", "Mean"]).lower()

        #### Section title
        st.subheader(f"You are currently checking the {measure} for the resource **{chosen_resource}**")

        #### Data extraction and prep
        resources_stats = md.resources_stats()

        stats = resources_stats.table(resources_df, resources_df.columns[:-1], "Origin")
        to_plot = resources_stats.to_plot(stats, [chosen_resource])

        translator = {"Total emissions" : "Total emissions (kg)",
                      "Land use per 1000kcal" : "Land use in squared meters (m2)",
                      "Land use per kg" : "Land use in squared meters (m2)",
                      "Land use per 100g protein" : "Land use in squared meters (m2)",
                      "Freshwater withdrawls per 1000kcal" : "Freshwater withdrawls in liters (l)",
                      "Freshwater withdrawls per kg" : "Freshwater withdrawls in liters (l)",
                      "Freshwater withdrawls per 100g protein" : "Freshwater withdrawls in liters (l)"}

        fig = px.bar(to_plot[to_plot["Measure"] == measure], x = "Resource", y = "Values", color = "Origin", color_discrete_map = {"Animal-based" : "red", "Plant-based" : "blue"}, barmode = "group", labels = {"Values" : translator[chosen_resource]})

        #### Data visualization
        st.write(fig)
        expander = st.beta_expander("Insights on the data")
        with expander:
            table = to_plot[to_plot["Measure"] == measure]
            st.table(table)


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
            di_object = md.daily_intake()
            daily_intake = di_object.get_data(daily_intake_df, gender, age)
            # daily_intake_object = md.daily_intake(gender, age)
            # daily_intake = daily_intake_object.get_data(daily_intake_df)

            # I get the series for each food group (as I need them that way later on) and save them in a list
            foods = [food_groups_stats[column] for column in food_groups_stats[food_groups].columns]

            # Make comparisons, using a comparator object
            comparator = md.comparator(foods, daily_intake)
            comparisons = comparator.get_comparisons()
            
            # Save the plot as a figure
            fig = vi.full_comparison_plot(comparisons)

            st.write(fig)
            st.table(comparisons[0].set_index("Food"))
    
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

        filter_by = st.sidebar.radio("Filter by:", options = ["Demographics", "Dietary", "Examination", "Laboratory", "Questionnaire"])

        # Table
        table_header = ["Variable name", "Variable description"]

        to_show = vardata.df.iloc[:, [0, 1, -2]].sort_values(by = translation[sort_by])
        to_show = to_show[to_show.component == filter_by]

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

        # Plots
        if button:
            # Data preprocessing
            features = [y] + X
            data = raw_health_df.loc[:, features]
            filtered_data = data.dropna()

            # Data description and some processing for later plotting
            data_stats = filtered_data.describe().T
            data_stats = data_stats.reset_index()
            data_stats = data_stats.drop("count", axis = 1)
            data_stats = data_stats.applymap(lambda x: md.round_number(x, 2))
            data_columns = list(data_stats.columns)

            # Correlations
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
                            columnwidth = [20, 10, 10, 10, 10, 10, 10, 10],
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

            # To adjust the height of the table and avoid as much as possible too much white space
            if len(features) < 6:
                table.update_layout(autosize = False, width = 600, height = 150,
                                    margin = dict(l = 0, r = 0, b = 0, t = 0))
            else:
                table.update_layout(autosize = False, width = 600, height = 200,
                                    margin = dict(l = 0, r = 0, b = 0, t = 0))

            ### Data insights
            # Expander
            expander = st.beta_expander("Insights on the data")
            with expander:
                st.write("**Chosen variables**:")
                for feature in features:
                    st.write(vardata.var_descr_detector(feature, nom_included = True))
                
                st.write("**Number of observations**:")
                st.write(f"- Before dropping the NaN values:\t{data.shape[0]}")
                st.write(f"- After dropping the NaN values:\t{filtered_data.shape[0]}")
                st.write("\n")
                st.write("**Target variable (y) values**:")
                st.table(filtered_data.loc[:, y].value_counts())

                nahnes_url = "https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017"
                st.write("More info in the following link:")
                st.markdown(nahnes_url, unsafe_allow_html=True)

            ### Correlation plot
            st.write(y_descr)
            colorscale = [[0, "white"], [1, "cornflowerblue"]]
            correlation_plot = ff.create_annotated_heatmap(corr,
                                                        #x = descrs,
                                                        y = descrs,
                                                        colorscale = colorscale)
            # Show the correlation plot
            st.write(correlation_plot)

            # Distribution plots for each chosen variable
            for x in X:
                x_descr = vardata.var_descr_detector(x, cut = 30, nom_included = True)
                expander = st.beta_expander(x_descr)

                # within expanders to ease the navigability
                with expander: 
                    to_plot = filtered_data.loc[:, [y, x]].dropna()
                    histogram = px.histogram(to_plot, x = x, color = y,
                                            marginal = "box",
                                            labels = {x : x_descr},
                                            width = 600)
                    st.write(histogram)

    if submenu == "Health Prediction":
        st.sidebar.write("Predict whether or not you can have a coronary disease")
        predict_button = st.sidebar.button("Predict health")

        # Form for the prediction
        expander = st.beta_expander("Find out if you are at risk of heart disease")

        with expander:
            cols = st.beta_columns(3)
            # Col 1
            GENDER = cols[0].text_input("Gender", value = "Female")
            if GENDER == "Female":
                FEMALE = 1
                MALE = 0
            else:
                FEMALE = 0
                MALE = 1
            RIDAGEYR = cols[0].text_input("Age", value = 43)
            BPXDI1 = cols[0].text_input("Diastolic: Blood pressure (mm Hg)", value = 68)
            BPXSY1 = cols[0].text_input("Systolic: Blood pressure (mm Hg)", value = 121) 
            BMXWT = cols[0].text_input("Weight (kg)", value = 79)

            # Col 2
            BMXWAIST = cols[1].text_input("Waist Circumference (cm)", value = 97)
            LBXTC = cols[1].text_input("Total Cholesterol (mg/dL) *", value = 183)
            LBXSGL = cols[1].text_input("Glucose (mg/dL) *", value = 100)
            MEANCHOL = cols[1].text_input("Cholesterol (gm) **", value = 290)
            MEANTFAT = cols[1].text_input("Total Fat (g) **", value = 78)

            # Col 3
            MEANSFAT = cols[2].text_input("Total Saturated Fatty Acis (g) **", value = 25)
            MEANSUGR = cols[2].text_input("Total Sugar (g) **", value = 103)
            MEANFIBE = cols[2].text_input("Total Fiber (g) **", value = 16)
            MEANTVB6 = cols[2].text_input("Total Vitamin B6 (mg) **", value = 2)

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
                st.subheader("You are at risk of having a coronary disease")
                st.write("This results are an estimation and you should always check with your doctor.")
            else:
                st.subheader("You are not at risk of having a coronary disease")
                st.write("This results are an estimation and you should always check with your doctor.")

####
if menu == "ML Models":
    st.header("Models without data scaling or balancing")
    st.write("These models were trained and tested using the raw data. This means, no scaling and no balancing of the data")
    st.table(models1_insights)

    st.header("Models with data scaling and balancing")
    st.write("These models were trained and tested using the modified data. This means, scaling and balancing the data")
    st.table(models2_insights)

    st.header("Conclusions")
    st.write("Although the models in the second table show lower scores, they reached higher recall levels, meaning that they were able to detect better positive cases of the coronary disease, which was the goal.\nFor this reason, the model used for the prediction section is the LogisticRegression with max_iter = 500 and warm start.")

    ### Correlation plot
    # st.write(y_descr)
    # colorscale = [[0, "white"], [1, "cornflowerblue"]]
    # correlation_plot = ff.create_annotated_heatmap(corr,
    #                                             #x = descrs,
    #                                             y = descrs,
    #                                             colorscale = colorscale)
    # # Show the correlation plot
    # st.write(correlation_plot)
    

####
if menu == "Glossary":
    #da.glossary()
    pass

####
if menu == "API":
    #da.api()
    pass