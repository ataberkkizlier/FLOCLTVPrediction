


##############################################################
# CLTV Prediction with BG-NBD and Gamma-Gamma
##############################################################

# Business Problem. FLO wants to define a roadmap for sales and
# # marketing activities. In order for the company to make a medium to long term plan, it is necessary to estimate the
# # potential value that existing customers will provide to the company in the future.



###############################################################
#Dataset Story
###############################################################

# The dataset consists of the past shopping behavior of customers who made their last purchases as OmniChannel (both online and offline shoppers) in 2020 - 2021.
# consists of the information obtained.

# master_id: Unique customer number
# order_channel : Which channel of the shopping platform is used (Android, iOS, Desktop, Mobile, Offline)
# last_order_channel : The channel where the last purchase was made
# first_order_date : Date of the customer's first purchase
# last_order_date : Date of the customer's last purchase
# last_order_date_online : The date of the last purchase made by the customer on the online platform
# last_order_date_offline : Date of the last purchase made by the customer on the offline platform
# order_num_total_ever_online : Total number of purchases made by the customer on the online platform
# order_num_total_ever_offline : Total number of purchases made by the customer offline
# customer_value_total_ever_offline : Total price paid by the customer for offline purchases
# customer_value_total_ever_online : Total price paid by the customer for online purchases
# interested_in_categories_12 : List of categories the customer has shopped in the last 12 months


###############################################################
# GOALS:
###############################################################
# TASK 1: Preparing the Data
           # 1. Read the flo_data_20K.csv data. Make a copy of the dataframe.
           # 2. Define the outlier_thresholds and replace_with_thresholds functions to suppress outliers.
           # Note: When calculating cltv, the frequency values must be integers, so round the lower and upper limits with round().
           # Define the variables "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online"
           # suppress any outliers.
           # 4. Omnichannel implies that customers shop both online and offline. Each customer's total
           # Create new variables for number of purchases and expenditure.
           # 5. Examine the types of variables. Change the type of variables that express date to date.

# TASK 2: Creating CLTV Data Structure
           # 1. Take 2 days after the date of the last purchase in the dataset as the analysis date.
           # 2.Create a new cltv dataframe with customer_id, recency_cltv_weekly, T_weekly, frequency and monetary_cltv_avg.
           # The monetary value will be expressed as the average value per purchase, and the recency and tenure values will be expressed in weekly terms.


# TASK 3: Establishment of BG/NBD, Gamma-Gamma Models, calculation of CLTV
           # 1. Fit the BG/NBD model.
                # a. Forecast expected purchases from customers in 3 months and add to the cltv dataframe as exp_sales_3_month.
                # b. Forecast the expected purchases from customers in 6 months and add to the cltv dataframe as exp_sales_6_month.
           # 2. Fit the Gamma-Gamma model. Estimate the average value that customers will leave and add it to the cltv dataframe as exp_average_value.
           # 3. Calculate the CLTV for 6 months and add it to the dataframe as cltv.
                # a. Standardize the calculated cltv values and create a scaled_cltv variable.
                # b. Observe the 20 people with the highest cltv values.

# TASK 4: Creating Segments According to CLTV
           # 1. Divide all your customers into 4 groups (segments) according to the 6-month standardized CLTV and add the group names to the dataset. Add them to the dataframe with the name cltv_segment.
           # 2. Make short 6-month action suggestions to the management for 2 groups you will choose among 4 groups

# TASK 5: Functionalize the whole process.


###############################################################
# TASK 1: Preparing the Data
# ###############################################################


import datetime as dt
import pandas as pd
#!pip install lifetimes
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.options.mode.chained_assignment = None

# 1. Read the flo_data_20K.csv data. Make a copy of the dataframe.
df_ = pd.read_csv("/Users/ataberk/Desktop/Miuul Bootcamp/week 3/FLOCLTVPrediction/flo_data_20k.csv")
df = df_.copy()
df.head()
df.describe().T
df.describe([0.01,0.25,0.5,0.75,0.99]).T

# 2. Define the outlier_thresholds and replace_with_thresholds functions to suppress outliers.
# Note: When calculating cltv, the frequency values must be integers, so round the lower and upper limits with round().
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit,0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)


# Variables "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online"
#suppress any outliers.
columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline","customer_value_total_ever_online"]
for col in columns:
    replace_with_thresholds(df, col)


# Which metrics do we need for CLTV?
# recency: Time between last purchase and first purchase. Weekly (user specific)
# T: Age of the customer. Weekly (how long before the date of analysis was the first purchase)
# frequency: total number of repeat purchases (frequency>1)
# monetary: average earnings per purchase



# 4. Omnichannel means that customers shop both online and offline.
# Create new variables for each customer's total number of purchases and spend.
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# 5. Examine the types of variables. Change the type of variables that express date to date.
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

###############################################################
# TASK 2: Creating CLTV Data Structure
###############################################################

# Take 2 days after the date of the last purchase in the dataset as the analysis date.
import datetime as dt

df["last_order_date"].max() # 2021-05-30
analysis_date = dt.datetime(2021,6,1)


# Create a new cltv dataframe with customer_id, recency_cltv_weekly, T_weekly, frequency and monetary_cltv_avg.
cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"]- df["first_order_date"]).dt.days) / 7
cltv_df["T_weekly"]= ((analysis_date- df["first_order_date"]).dt.days) /7
cltv_df["frequency"] = df["order_num_total"]
cltv_df["monetary_cltv_avg"] = df["customer_value_total"] / df["order_num_total"]


cltv_df.head()

###############################################################
# TASK 3: Establishment of BG/NBD, Gamma-Gamma Models, calculation of 6-month CLTV
###############################################################

# 1. Set up the BG/NBD model.
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])

# Estimate the expected purchases from customers in 3 months and add it to the cltv dataframe as exp_sales_3_month.
cltv_df["exp_sales_3_month"] = bgf.predict(4*3,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])


cltv_df.sort_values("exp_sales_3_month", ascending=False).head()


# Estimate the expected purchases from customers in 6 months and add it to the cltv dataframe as exp_sales_6_month.
cltv_df["exp_sales_6_month"] = bgf.predict(4*6,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])

cltv_df.sort_values("exp_sales_6_month", ascending=False).head(10)

# Analyze the 10 people who will make the most purchases in the 3rd and 6th month. Is there a difference?
cltv_df.sort_values("exp_sales_3_month",ascending=False)[:10]

cltv_df.sort_values("exp_sales_6_month",ascending=False)[:10]


# 2. Fit the Gamma-Gamma model. Predict the average value that customers will leave and add it to the cltv dataframe as exp_average_value.
# We need the recency and monetary metrics.

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
cltv_df.head()
# 3. Calculate the CLTV for 6 months and add it to the dataframe with the name cltv.
# The customer_lifetime_value function asks for the gamma-gamma model and the bgnbd model.
# Which metrics are we going to calculate on?

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6, #aylık
                                   freq="W", # verilen bilgiler aylık mı? haftalık mı?
                                   discount_rate=0.01) #kampanya etkisi göz önünde bulunduruluyor.
cltv_df["cltv"] = cltv

cltv_df.head()


# Observe the 20 people with the highest CLTV values.
cltv_df.sort_values("cltv",ascending=False)[:20]

###############################################################
# TASK 4: Creating Segments According to CLTV
###############################################################

# 1. Divide all your customers into 4 groups (segments) according to the 6-month standardized CLTV and add the group names to the data set.
# Assign them with the name cltv_segment.

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.head()


# 2. # Does it make sense to categorize customers into 4 groups according to their CLTV scores? Should it be more or less.
cltv_df.groupby("cltv_segment").agg({"count","mean","sum"})

cltv_df[["cltv_segment", "recency_cltv_weekly", "frequency", "monetary_cltv_avg"]].groupby("cltv_segment").agg(["mean", "count"])



# Make short 6-month action proposals to the management for 2 groups you will choose from among 4 groups.



###############################################################
# BONUS:
###############################################################

def create_cltv_df(dataframe):

    # Setting up the data
    columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline","customer_value_total_ever_online"]
    for col in columns:
        replace_with_thresholds(dataframe, col)

    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    dataframe = dataframe[~(dataframe["customer_value_total"] == 0) | (dataframe["order_num_total"] == 0)]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)

    # Creation of CLTV data structure
    dataframe["last_order_date"].max()  # 2021-05-30
    analysis_date = dt.datetime(2021, 6, 1)
    cltv_df = pd.DataFrame()
    cltv_df["customer_id"] = dataframe["master_id"]
    cltv_df["recency_cltv_weekly"] = ((dataframe["last_order_date"] - dataframe["first_order_date"]).astype('timedelta64[D]')) / 7
    cltv_df["T_weekly"] = ((analysis_date - dataframe["first_order_date"]).astype('timedelta64[D]')) / 7
    cltv_df["frequency"] = dataframe["order_num_total"]
    cltv_df["monetary_cltv_avg"] = dataframe["customer_value_total"] / dataframe["order_num_total"]
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

    # Setting the BG-NBD Model
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])
    cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])
    cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])

    plot_period_transactions(bgf)
    plt.show(block=True)

    # # Setting the Gamma-Gamma Model
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
    cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                           cltv_df['monetary_cltv_avg'])

    # Cltv prediction
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'],
                                       cltv_df['monetary_cltv_avg'],
                                       time=6,
                                       freq="W",
                                       discount_rate=0.01)
    cltv_df["cltv"] = cltv

    # CLTV segmentation
    cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_df

cltv_df = create_cltv_df(df)


cltv_df.head(20)


