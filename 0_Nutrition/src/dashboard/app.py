import streamlit as st

import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

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

    # Load data
    resources_df = pd.read_csv(environment_data_path + "resources.csv", index_col = 0)
    nutrition_df = pd.read_csv(environment_data_path + "nutritional_values.csv", index_col = 0)
    daily_intake_df = pd.read_csv(environment_data_path + "daily_intakes.csv")
    
    return resources_df, nutrition_df, daily_intake_df

resources_df, nutrition_df, daily_intake_df = get_data()


##################################################### INTERFACE #####################################################
####
menu = st.sidebar.selectbox("Menu:",
                            options = ["Home", "Resources Facts", "Nutrition Facts"])

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

        color_map = md.color_mapper(selection)
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

    #### Filters
    st.subheader("You can filter the data using the checkboxes")
    cols = st.beta_columns(2)

    cols[0].write("Positive filters")
    positive_filters = ["Milks", "Cheese", "Other Animal Products", "Meats", "Chicken", "Fish",
                        "Milk Substitutes", "Beans", "Soy Products", "Nuts", "Other Veggie Products"]

    positive_checkboxes = []
    for filter_ in positive_filters:
        checkbox = cols[0].checkbox(filter_)
        if checkbox:
            positive_checkboxes.append(filter_)

    cols[1].write("Negative filters")
    negative_filters = ["Others", "Baby Food", "Desserts And Snacks", "Drinks", "Sandwiches", "Prepared Dishes", "Sauces"]

    negative_checkboxes = []
    for filter_ in negative_filters:
        checkbox = cols[1].checkbox(filter_)
        if checkbox:
            negative_checkboxes.append(filter_)

    filtered_df = nutrition_df
    filter_tool = md.filter_tool

    if len(positive_checkboxes) > 0:
        f_ = filter_tool.multiple_filter(positive_checkboxes)
        filtered_df = filter_tool.rows_selector(filtered_df, f_)
    
    st.write(filtered_df.shape)




    # cols[0].checkbox("Milks")
    # cols[0].checkbox("Cheese")
    # cols[0].checkbox("Other Animal Products")
    # cols[0].checkbox("Meats")
    # cols[0].checkbox("Chicken")
    # cols[0].checkbox("Fish")
    # cols[0].checkbox("Milk Substitutes")
    # cols[0].checkbox("Beans")
    # cols[0].checkbox("Soy Products")
    # cols[0].checkbox("Nuts")
    # cols[0].checkbox("Other Veggie Products")


    # To choose between subsections
    submenu = st.sidebar.radio(label = "What do you want to do?", options = ["Top products", "Food groups", "Foods"])
    
    # Subsection 1
    if submenu == "Top products":
        # User input
        chosen_nutrient = st.sidebar.selectbox("Nutrient", options = nutrition_df.columns[3:-2])
        entries = st.sidebar.slider(label = "How many foods?", min_value = 5, max_value = 50, 
                                    value = 5, step = 5)

        filter_tool = md.filter_tool
        table = filter_tool.nutrient_selector(nutrition_df, chosen_nutrient).head(entries)

        st.write(table)