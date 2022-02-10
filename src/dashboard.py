# import libraries
import os
import sys
from typing import OrderedDict
import numpy as np
import json
import pandas as pd

import streamlit as st
from datetime import datetime
from lifetimes import BetaGeoFitter

sys.path.insert(0, "..")

from src.plotting_utils import plot_proba_alive, plot_purchases_history


DATA_FOLDER_PATH = 'data'
DF_ORDER_PATH = os.path.join(DATA_FOLDER_PATH, 'df_order.csv')
CUSTOMERS_CHAR_PATH = os.path.join(DATA_FOLDER_PATH, 'customers_characteristics.csv')




class DashboardApp:
    """Class to generate a streamlit app combined all required graphs in 4 pages
    """
    batch_id = 0
    def __init__(self):
        # TODO
        pass

    @st.cache()
    def load_data(self):
        df_order = pd.read_csv(DF_ORDER_PATH)
        df_order.date_created = pd.to_datetime(df_order.date_created)
        customers_characteristics = pd.read_csv(CUSTOMERS_CHAR_PATH, index_col="customer_id")
        customers_characteristics.date_created_min = pd.to_datetime(customers_characteristics.date_created_min)
        customers_characteristics.date_created_max = pd.to_datetime(customers_characteristics.date_created_max)
        return df_order, customers_characteristics


    def configure_page(self):
        """
        Configures app page
        Creates sidebar with selectbox leading to different main pages
        Returns:
            option (str): Name of main page selected by user
        """
        st.set_page_config(
            page_title="Churn prediction",
            layout="wide",
            initial_sidebar_state="expanded"
        )



        # create sidebar
        st.sidebar.title("Parameters")
        self.df_order, self.customers_characteristics = self.load_data()
        self.start_time = st.sidebar.slider(
            "Last purchase after:",
            min_value=datetime(2017, 9, 22),
            max_value=datetime(2019, 9, 22),
            value=datetime(2019, 6, 25),
            format="YYYY-MM-DD")

        sorted_by_risk_df = self.customers_characteristics[
                (self.customers_characteristics.frequency>20)&
                (self.customers_characteristics.date_created_max>self.start_time)]\
                .sort_values(by="p_loss", ascending=False)
        
        sorted_by_risk_df["monthly_revenues"] = sorted_by_risk_df["monthly_revenues"].astype(int)
        colored_df = sorted_by_risk_df[["monthly_revenues", "proba"]]
        colored_df["proba"] = 1-colored_df["proba"]
        colored_df.columns = ["Month. exp.", "Proba churn"]
        colored_df = colored_df.style.background_gradient(cmap='RdYlGn_r',
                        vmin=0,
                        vmax=1,
                        subset=["Proba churn"])
        st.sidebar.subheader("Key clients to target:")
        with st.sidebar.expander("See explanation"):
            st.write("""
                A client is considered as key to target when his monthly expenses
                are high and he is likely to be churning.
            """)
        st.sidebar.dataframe(colored_df)

        self.customer_id = st.sidebar.selectbox(
            options = sorted_by_risk_df.index.tolist(),
            label = "Select customer to deep dive"
        )

    @st.cache()
    def fit_model(self):
        bgf = BetaGeoFitter(penalizer_coef=0)
        bgf.fit(self.customers_characteristics['frequency'],
            self.customers_characteristics['recency'], 
            self.customers_characteristics['T'])
        return bgf

    def create_main_pages(self):
        """
        Creates pages for all options available in sidebar
        Page is loaded according to option picked in sidebar
        """
        # Main Dashboard
        st.title("Churn prediction")

        st.subheader(f"Customer {self.customer_id} info:")
        cltv = self.customers_characteristics.loc[self.customer_id, "total_revenue_sum"]
        mean_cltv = self.customers_characteristics["total_revenue_sum"].mean()
        monthly_rev = self.customers_characteristics.loc[self.customer_id, "monthly_revenues"]
        mean_monthly = self.customers_characteristics["monthly_revenues"].mean()
        age = self.customers_characteristics.loc[self.customer_id, "T"]
        mean_age = self.customers_characteristics["T"].mean()
        freq = self.customers_characteristics.loc[self.customer_id, "frequency"]
        mean_freq = self.customers_characteristics["frequency"].mean()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("CLTV", "{:,}".format(round(cltv))+"$", str(round(((cltv-mean_cltv)/mean_cltv), 2))+"%*")
        col2.metric("Monthly expenses when active", "{:,}".format(round(monthly_rev))+"$", str(round(((monthly_rev-mean_monthly)/mean_monthly), 2))+"%*")
        col3.metric("Age", str(round(age))+" days", "{:,}".format(round(((age-mean_age)/mean_age), 2))+"%*")
        col4.metric("Number of transactions", round(freq)+1, "{:,}".format(round(((freq-mean_freq)/mean_freq), 2))+"%*")
        st.write("*vs. average")

        st.subheader(f"Probability that the customer {self.customer_id} is still active over time:")
        bgf = self.fit_model()
        fig = plot_proba_alive(self.customer_id, self.df_order, self.customers_characteristics, bgf)
        st.pyplot(fig)

        st.subheader(f"Transaction history of customer {self.customer_id}:")
        fig2 = plot_purchases_history(self.df_order, self.customer_id)
        st.plotly_chart(fig2)



        

