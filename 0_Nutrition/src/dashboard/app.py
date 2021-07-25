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
    nutrition_df = pd.read_csv(environment_data_path + "nutritional_values.csv")
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
    st.sidebar.subheader("Play around")

    #### Subsection 1
    if submenu == "Resources facts":
        # User input
        chosen_resource = st.sidebar.selectbox('Choose a resource:', options = resources_df.columns)
        entries = st.sidebar.slider(label = "Entries:", min_value = 10,
                                    max_value = 50, value = 10, step = 10)

        # Page title
        st.title("This is the resources facts section")
        st.subheader(f"You are currently checking the top {entries} by **{chosen_resource}**")

        # Data filtering
        selection = resources_df[chosen_resource].sort_values(ascending = False).head(entries)
        
        # Creating table/plots
        header = ["Food", selection.name]
        data = pd.DataFrame([selection.index, selection.values])

        table = go.Figure(data = go.Table(
                            columnwidth = [70, 30],
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

        fig = px.bar(selection, x = selection.index, y = chosen_resource)

        # Data visualization
        st.write(fig)
        st.write(table)
        #st.table(selection)

    pass

####
if menu == "Nutrition Facts":
    #da.nutrition_facts()
    pass