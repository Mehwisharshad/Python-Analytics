#!/usr/bin/env python
# coding: utf-8

# #### IMPORTING LIBRARIES

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import plotly.express as px
import matplotlib.pyplot as plt
import time
import os
import warnings
import shap
import plotly.graph_objects as go

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
from collections import Counter
from datetime import datetime  
from plotly.subplots import make_subplots
import nbformat

warnings.filterwarnings("ignore")                       # Suppress all warnings


# #### CONTROL PARAMETERS

# In[3]:


DATE_COL = 'PLC_DATE_TIME'                                      
KPI_TO_ANALYZE = 'U_16_ACTIVE_POWER_TOTAL_KW'                   # Also check U_27_ACTIVE_ENERGY_IMPORT_KWH, U_27_ACTIVE_POWER_TOTAL_KW,U_23_ACTIVE_POWER_TOTAL_KW
z_score_threshold = 2
DT_filter = "Date >= '2023-03-01'"

#DT_filter = "Date >= '2022-10-22'"
#DT_filter = "Date >= '2022-10-22' and Hour == 13"              # Define your criteria


# #### READING FILE

# In[4]:


file_path = 'C:\\xampp\\htdocs\\Analytics\\Output_file.csv'
df = pd.read_csv(file_path)                                   # Load JSON file into a DataFrame
#df.head(5)


# #### DATA CLEANING

# In[5]:


df = df.drop(columns=['_id', 'timestamp', 'UNIXtimestamp', 'SCRAP_11', '_msgid','Time'])          # Dropping the specified columns
#df.head(5)


# In[6]:


cols_to_drop = df.filter(regex='^U_1_').columns                 # Filter columns that start with 'U_1_'
df = df.drop(columns=cols_to_drop)                              # Drop the filtered columns
#df.head(5)


# #### EXTRACTING DATE AND TIME

# In[7]:


df['PLC_DATE_TIME'] = df['PLC_DATE_TIME'].str.replace('DT#', '')
df['Date'] = pd.to_datetime(df['PLC_DATE_TIME'].str[:10])
df['Hour'] = pd.to_datetime(df['PLC_DATE_TIME'].str[-8:]).dt.round('h').dt.hour


# #### FILTER REQUIRED DATES

# In[8]:


filtered_df = df.query(DT_filter)                                       # Filter data for dates >= '2022-10-22'

# Other Alternatives
#filtered_df = df[df['Date'] >= '2022-10-22']
#filtered_df = df[(df['Date'] >= '2022-10-22') & (df['Hour'] == 12)]


# #### REORDERING COLUMNS

# In[9]:


# Reorder columns to place 'Date' and 'Hour' at the start
new_column_order = ['Date', 'Hour'] + [col for col in filtered_df.columns if col not in ['Date', 'Hour']]
filtered_df = filtered_df[new_column_order]
filtered_df.head(5)


# #### DEALING WITH MISSING VALUES

# In[10]:


missing_percentage = filtered_df.isnull().mean() * 100          # Calculate the percentage of missing values for each column
# Sort the results in descending order
missing_percentage_sorted = missing_percentage.sort_values(ascending=False)
missing_percentage_sorted.head(10)


# In[11]:


#filtered_df = filtered_df.ffill()                  # Fill missing values using forward fill
filtered_df = filtered_df.fillna(0)                 # Fill missing values with 0s
#filtered_df.head(5)


# #### CORRECTING DATA TYPES

# In[12]:


# Convert 'PLC_DATE_TIME' column to datetime
filtered_df['PLC_DATE_TIME'] = pd.to_datetime(filtered_df['PLC_DATE_TIME'])

# Verify the conversion
print(filtered_df['PLC_DATE_TIME'].dtypes)


# #### REQUIRED ASSIGNMENTS

# In[13]:


X = filtered_df[[KPI_TO_ANALYZE]]                         # Assuming 'KPI_TO_ANALYZE' is the column name in df that you want to analyze


# In[14]:


filtered_df.head(2)


# ## DAY BASED COMPARISONS

# #### DAY VS. BUSIEST SAME DAY

# In[15]:


# Define the day of the week you want to analyze
day_of_week_to_analyze = 'Monday'


# In[16]:


# Step 1: Append day name against each date
filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])
filtered_df['Day Name'] = filtered_df['Date'].dt.day_name()

# Convert 'Hour' from number to time
filtered_df['Hour'] = pd.to_timedelta(filtered_df['Hour'], unit='h')
filtered_df['Hour'] = filtered_df['Hour'].dt.components.hours  # Keeping only the hour component

# Find the last available day_of_week_to_analyze
last_day_date = filtered_df[filtered_df['Day Name'] == day_of_week_to_analyze]['Date'].max()
last_day_data = filtered_df[(filtered_df['Day Name'] == day_of_week_to_analyze) & (filtered_df['Date'] == last_day_date)]

# Find the busiest day_of_week_to_analyze based on the total KPI value
days_grouped = filtered_df[filtered_df['Day Name'] == day_of_week_to_analyze].groupby('Date')[KPI_TO_ANALYZE].sum()
busiest_day_date = days_grouped.idxmax()
busiest_day_data = filtered_df[(filtered_df['Day Name'] == day_of_week_to_analyze) & (filtered_df['Date'] == busiest_day_date)]


# In[17]:


# Plot the data
#fig = px.line(title=f'HOURLY VALUE of {KPI_TO_ANALYZE} FOR LAST VS. BUSIEST {day_of_week_to_analyze}')
fig = px.line(title=f'HOURLY VALUE of {KPI_TO_ANALYZE} <br>LAST VS. BUSIEST {day_of_week_to_analyze.upper()}')
fig.add_scatter(
    x=last_day_data['Hour'], 
    y=last_day_data[KPI_TO_ANALYZE], 
    mode='lines', 
    line_shape='spline', 
    name=f'Last {day_of_week_to_analyze} ({last_day_date.date()})'
)
fig.add_scatter(
    x=busiest_day_data['Hour'], 
    y=busiest_day_data[KPI_TO_ANALYZE], 
    mode='lines', 
    line_shape='spline', 
    name=f'Busiest {day_of_week_to_analyze} ({busiest_day_date.date()})'
)
fig.update_layout(
    xaxis_title='Hour', 
    yaxis_title=KPI_TO_ANALYZE, 
    xaxis=dict(tickmode='linear', tick0=0, dtick=1, showgrid=True), 
    yaxis=dict(showgrid=True), 
    plot_bgcolor='rgba(0,0,0,0)', 
    paper_bgcolor='white', 
    width=1000, 
    height=500,
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=-0.3,
        xanchor='center',
        x=0.5
    )
)
fig.show()


# #### COMPARE DAY WITH PREVIOUS SAME DAYS

# In[18]:


# Append day name against each date
filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])
filtered_df['Day Name'] = filtered_df['Date'].dt.day_name()

# Convert 'Hour' from number to time
filtered_df['Hour'] = pd.to_timedelta(filtered_df['Hour'], unit='h')
filtered_df['Hour'] = filtered_df['Hour'].dt.components.hours  # Keeping only the hour component

# Find the last available day of the week to analyze
last_day_date = filtered_df[filtered_df['Day Name'] == day_of_week_to_analyze]['Date'].max()

# Get the previous three days of the week to analyze
all_days = filtered_df[filtered_df['Day Name'] == day_of_week_to_analyze]['Date'].drop_duplicates().sort_values(ascending=False)
previous_days = all_days[all_days < last_day_date].head(3).sort_values()

# Combine the last day with the previous three days
days_to_plot = pd.concat([previous_days, pd.Series([last_day_date])]).sort_values()

# Plot the data for each day
fig = px.line(title=f'HOURLY VALUE OF {KPI_TO_ANALYZE} <br>LAST & PREVIOUS 3 {day_of_week_to_analyze.upper()}s')

for day_date in days_to_plot:
    day_data = filtered_df[(filtered_df['Day Name'] == day_of_week_to_analyze) & (filtered_df['Date'] == day_date)]
    fig.add_scatter(
        x=day_data['Hour'], 
        y=day_data[KPI_TO_ANALYZE], 
        mode='lines', 
        line_shape='spline', 
        name=f'{day_of_week_to_analyze} ({day_date.date()})'
    )


# In[19]:


fig.update_layout(
    xaxis_title='Hour', 
    yaxis_title=KPI_TO_ANALYZE, 
    xaxis=dict(tickmode='linear', tick0=0, dtick=1, showgrid=True), 
    yaxis=dict(showgrid=True), 
    plot_bgcolor='rgba(0,0,0,0)', 
    paper_bgcolor='white', 
    width=1000, 
    height=500,
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=-0.3,
        xanchor='center',
        x=0.5
    )
)

fig.show()


# ## DENSITY PLOTS

# In[20]:


# Create density traces for each day of the week

days_of_week = filtered_df['Day Name'].unique()
fig = go.Figure()

for day in days_of_week:
    day_data = filtered_df[filtered_df['Day Name'] == day][KPI_TO_ANALYZE]
    kde = gaussian_kde(day_data, bw_method=0.3)  # Adjust bandwidth as needed
    x_vals = np.linspace(day_data.min(), day_data.max(), 1000)
    y_vals = kde(x_vals)
    
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        name=day
    ))

# fig = px.line(title=f'HOURLY VALUE OF {KPI_TO_ANALYZE} <br>LAST & PREVIOUS 3 {day_of_week_to_analyze.upper()}s')

fig.update_layout(
    #title='PROBABILITY DENSITY PLOTS OF KPI_TO_ANALYZE FOR DIFFERENT DAYS',
    title=f'PROBABILITY DENSITY PLOTS OF {KPI_TO_ANALYZE} <br>DIFFERENT DAYS OF WEEK',
    xaxis_title=KPI_TO_ANALYZE,
    yaxis_title='Density',
    legend_title='Day Name',
    width=1000,
    height=500,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='white',
    xaxis=dict(showgrid=True),
    yaxis=dict(showgrid=True)
)

fig.show()


# In[21]:


fig = px.histogram(filtered_df, x=KPI_TO_ANALYZE, color='Day Name', facet_col='Day Name', facet_col_wrap=3,
                   histnorm='probability density', width=1000, height=500, 
                   #title='PROBABILITY DENSITY PLOTS OF KPI_TO_ANALYZE FOR DIFFERENT DAYS OF WEEK'
                   title=f'PROBABILITY DENSITY PLOTS OF {KPI_TO_ANALYZE} <br>DIFFERENT DAYS OF WEEK')
#fig.update_traces(marker=dict(size=0))
fig.update_layout(plot_bgcolor='white')
fig.show()


# ## HOUR BASED ANALYSIS

# #### AVERAGE VALUE PER WEEKDAY

# In[22]:


# Calculate the average 'KPI_TO_ANALYZE' for each hour for each day
average_kpi_per_hour_day = filtered_df.groupby(['Day Name', 'Hour'])[KPI_TO_ANALYZE].mean().reset_index()
#average_kpi_per_hour_day = filtered_df.groupby(['Day Name', 'Hour'])[KPI_TO_ANALYZE].sum().reset_index()

# Interpolate using cubic spline
fig = px.line()
for day in average_kpi_per_hour_day['Day Name'].unique():
    day_data = average_kpi_per_hour_day[average_kpi_per_hour_day['Day Name'] == day]
    f = interp1d(day_data['Hour'], day_data[KPI_TO_ANALYZE], kind='cubic')
    x_new = np.linspace(day_data['Hour'].min(), day_data['Hour'].max(), 500)
    fig.add_scatter(x=x_new, y=f(x_new), mode='lines', name=day)

fig.update_layout(
    xaxis_title='Hour',
    yaxis_title='Average KPI_TO_ANALYZE',
    #title='Average KPI_TO_ANALYZE by Hour for Each Day',
    title=f'AVERAGE {KPI_TO_ANALYZE} BY HOUR<br>SEPARATE FOR EACH DAY',
    xaxis=dict(tickmode='linear', dtick=1),  # Ensure x-axis has all hours from 0 to 23
    legend_title='Day Name',
    plot_bgcolor='white',
    width=1000,
    height=500
)
fig.show()


# #### AVERAGE POWER PER HOUR

# In[23]:


# Calculate the average KPI value for each hour
average_kpi_per_hour = filtered_df.groupby('Hour')[KPI_TO_ANALYZE].mean().reset_index()

# Assuming 'average_kpi_per_hour' DataFrame is already defined
import plotly.express as px

fig = px.bar(average_kpi_per_hour, x='Hour', y=KPI_TO_ANALYZE, 
             width=1000, height=500,
             title=f'AVERAGE {KPI_TO_ANALYZE} <br>PER HOUR AVERAGE OF ALL DAYS')

fig.update_layout(
    xaxis_title='Hour',
    yaxis_title='Average KPI_TO_ANALYZE',
    plot_bgcolor='white',
    showlegend=False
)

fig.show()


# #### TOTAL POWER PER HOUR

# In[24]:


# Calculate the average KPI value for each hour
average_kpi_per_hour = filtered_df.groupby('Hour')[KPI_TO_ANALYZE].sum().reset_index()

# Assuming 'average_kpi_per_hour' DataFrame is already defined
import plotly.express as px

fig = px.bar(average_kpi_per_hour, x='Hour', y=KPI_TO_ANALYZE, 
             width=1000, height=500,
             title=f'TOTAL ENERGY CONSUMED BY {KPI_TO_ANALYZE} <br>PER HOUR SUM OF ALL DAYS')

fig.update_layout(
    xaxis_title='Hour',
    yaxis_title='Average KPI_TO_ANALYZE',
    plot_bgcolor='white',
    showlegend=False
)

fig.show()


# ## DAY LEVEL ANALYSIS

# #### AVERAGE POWER PER WEEK DAY

# In[25]:


# Append day name against each date
filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])
filtered_df['Day Name'] = filtered_df['Date'].dt.day_name()

# Calculate the average value of KPI_TO_ANALYZE for each day of the week
average_kpi_by_day = filtered_df.groupby('Day Name')[KPI_TO_ANALYZE].mean().reset_index()

# Ensure the days are in the correct order
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
average_kpi_by_day['Day Name'] = pd.Categorical(average_kpi_by_day['Day Name'], categories=day_order, ordered=True)
average_kpi_by_day = average_kpi_by_day.sort_values('Day Name')

# Plot the data as a bar chart
fig = px.bar(
    average_kpi_by_day,
    x='Day Name',
    y=KPI_TO_ANALYZE,
    title=f'AVERAGE {KPI_TO_ANALYZE} <br>BY DAY OF WEEK'
)

fig.update_layout(
    xaxis_title='Day of the Week',
    yaxis_title=f'Average {KPI_TO_ANALYZE}',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='white',
    width=1000,
    height=500,
    xaxis=dict(
        tickmode='linear',
        showgrid=True
    ),
    yaxis=dict(
        showgrid=True
    ),
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=-0.3,
        xanchor='center',
        x=0.5
    )
)

fig.show()


# #### TOTAL POWER PER DAY

# In[26]:


# Convert 'Date' column to datetime
filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])

# Calculate the sum of KPI value for each date
sum_kpi_per_date = filtered_df.groupby('Date')[KPI_TO_ANALYZE].sum().reset_index()

# Calculate the average KPI value
average_kpi_value = sum_kpi_per_date[KPI_TO_ANALYZE].mean()

# Plot the bar chart
fig = px.bar(sum_kpi_per_date, x='Date', y=KPI_TO_ANALYZE, 
             width=1000, height=500,
             title=f'TOTAL ENERGY CONSUMED BY {KPI_TO_ANALYZE} <br>BY DATE')

# Add a threshold line for the average KPI value
fig.add_shape(
    type='line',
    x0=sum_kpi_per_date['Date'].min(), 
    y0=average_kpi_value, 
    x1=sum_kpi_per_date['Date'].max(), 
    y1=average_kpi_value,
    line=dict(color='Red', dash='dash'),
    name='Average'
)

fig.update_layout(
    xaxis_title='Date',
    yaxis_title=f'TOTAL ENERGY UTILIZED {KPI_TO_ANALYZE}',
    plot_bgcolor='white',
    showlegend=False
)

# Add annotation for the threshold line
fig.add_annotation(
    x=sum_kpi_per_date['Date'].max(), 
    y=average_kpi_value,
    text=f'Average: {average_kpi_value:.2f}',
    showarrow=False,
    yshift=10
)

fig.show()


# #### TOTAL POWER PER WEEK

# In[27]:


# Convert 'Date' column to datetime
filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])

# Extract week number from the Date column
filtered_df['Week'] = filtered_df['Date'].dt.isocalendar().week

# Calculate the sum of KPI values for each week
sum_kpi_per_week = filtered_df.groupby('Week')[KPI_TO_ANALYZE].sum().reset_index()

# Calculate the average KPI value
average_kpi_value = sum_kpi_per_week[KPI_TO_ANALYZE].mean()

# Plot the bar chart
fig = px.bar(sum_kpi_per_week, x='Week', y=KPI_TO_ANALYZE, 
             width=1000, height=500,
             title=f'TOTAL ENERGY CONSUMED BY {KPI_TO_ANALYZE} <br>BY WEEK')

# Add a threshold line for the average KPI value
fig.add_shape(
    type='line',
    x0=sum_kpi_per_week['Week'].min(), 
    y0=average_kpi_value, 
    x1=sum_kpi_per_week['Week'].max(), 
    y1=average_kpi_value,
    line=dict(color='Red', dash='dash'),
    name='Average'
)

fig.update_layout(
    xaxis_title='Week',
    yaxis_title=f'TOTAL ENERGY UTILIZED {KPI_TO_ANALYZE}',
    plot_bgcolor='white',
    showlegend=False
)

# Add annotation for the threshold line
fig.add_annotation(
    x=sum_kpi_per_week['Week'].max(), 
    y=average_kpi_value,
    text=f'Average: {average_kpi_value:.2f}',
    showarrow=False,
    yshift=10
)

fig.show()


# #### TOTAL POWER PER MONTH

# In[28]:


# Convert 'Date' column to datetime
filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])

# Extract month and year from the Date column
filtered_df['Month_Year'] = filtered_df['Date'].dt.strftime('%B %Y')

# Calculate the sum of KPI values for each month and year
sum_kpi_per_month_year = filtered_df.groupby('Month_Year')[KPI_TO_ANALYZE].sum().reset_index()

# Ensure the Month_Year column is sorted by date
sum_kpi_per_month_year['Date'] = pd.to_datetime(sum_kpi_per_month_year['Month_Year'], format='%B %Y')
sum_kpi_per_month_year = sum_kpi_per_month_year.sort_values('Date')

# Plot the bar chart
fig = px.bar(sum_kpi_per_month_year, x='Month_Year', y=KPI_TO_ANALYZE, 
             width=1000, height=500,
             title=f'TOTAL ENERGY CONSUMED BY {KPI_TO_ANALYZE} <br>BY MONTH AND YEAR')

fig.update_layout(
    xaxis_title='Month and Year',
    yaxis_title=f'TOTAL ENERGY UTILIZED {KPI_TO_ANALYZE}',
    plot_bgcolor='white',
    showlegend=False
)

fig.show()


# ## OPERATING HOURS

# In[29]:


#--------------
#Standard Hours
#--------------

# Filter the data where KPI_TO_ANALYZE > 0
standard_hours = filtered_df[filtered_df[KPI_TO_ANALYZE] > 10]

# Group by 'Hour' and count the occurrences of hours
hour_counts = standard_hours.groupby('Hour').size().reset_index(name='Count')

# Plot the count of hours having KPI_TO_ANALYZE > 0 using Plotly Express
fig = px.bar(hour_counts, x='Hour', y='Count', title=f'COUNT OF INSTANCES HAVING {KPI_TO_ANALYZE} > 0 <br>PER HOUR')
fig.update_xaxes(title='Hour')
fig.update_yaxes(title='Count')

# Adjust the size of the chart
fig.update_layout(
    width=1000,  # Set the width of the chart
    height=500,  # Set the height of the chart
)

fig.show()


# #### WEEKWISE OPERATING HOURS

# In[30]:


#-----------------------------
# Operating Hours_Last 4 weeks
#-----------------------------

# Filter the data where KPI_TO_ANALYZE > 0
standard_hours = filtered_df[filtered_df[KPI_TO_ANALYZE] > 10]

# Get the last 4 weeks
last_4_weeks = sorted(standard_hours['Date'].dt.isocalendar().week.unique())[-4:]

# Filter data for the last 4 weeks
standard_hours_last_4_weeks = standard_hours[standard_hours['Date'].dt.isocalendar().week.isin(last_4_weeks)]

# Group by 'Week' and 'Hour' and count the occurrences of hours
hour_counts = standard_hours_last_4_weeks.groupby(['Week', 'Hour']).size().reset_index(name='Count')

# Plot the count of hours having KPI_TO_ANALYZE > 0 using Plotly Express
fig = px.bar(hour_counts, x='Hour', y='Count', color='Week', barmode='group', title=f'COUNT OF INSTANCES HAVING {KPI_TO_ANALYZE} > 0<br>WEEK WISE FOR LAST 4 WEEKS')
fig.update_xaxes(title='Hour')
fig.update_yaxes(title='Count')

# Adjust the size of the chart
fig.update_layout(
    width=1000,  # Set the width of the chart
    height=500,  # Set the height of the chart
)

fig.show()


# In[31]:


# Filter the data where KPI_TO_ANALYZE > 0
standard_hours = filtered_df[filtered_df[KPI_TO_ANALYZE] > 10]

# Get the last 4 weeks
last_4_weeks = sorted(standard_hours['Date'].dt.isocalendar().week.unique())[-4:]

# Filter data for the last 4 weeks
standard_hours_last_4_weeks = standard_hours[standard_hours['Date'].dt.isocalendar().week.isin(last_4_weeks)]

# Group by 'Week' and 'Hour' and count the occurrences of hours
hour_counts = standard_hours_last_4_weeks.groupby(['Week', 'Hour']).size().reset_index(name='Count')

# Plot the count of hours having KPI_TO_ANALYZE > 0 using Plotly Express
fig = px.bar(hour_counts, x='Hour', y='Count', color='Week', barmode='group', title=f'COUNT OF INSTANCES HAVING {KPI_TO_ANALYZE} > 0<br>WEEK WISE FOR LAST 4 WEEKS', facet_row='Week')
fig.update_xaxes(title='Hour')
fig.update_yaxes(title='Count')

# Adjust the gap width between bars
fig.update_layout(bargap=0.005)  # Adjust the value as needed

# Adjust the size of the chart
fig.update_layout(
    width=1000,  # Set the width of the chart
    height=500,  # Set the height of the chart
)

fig.show()


# In[32]:


# Filter the data where KPI_TO_ANALYZE > 0
#standard_hours = filtered_df[filtered_df[KPI_TO_ANALYZE] > 10]

# Get the last 4 weeks
#standard_hours['Week'] = standard_hours['Date'].dt.isocalendar().week
#last_4_weeks = sorted(standard_hours['Week'].unique())[-4:]

# Filter data for the last 4 weeks
#standard_hours_last_4_weeks = standard_hours[standard_hours['Week'].isin(last_4_weeks)]

# Create a new column combining 'Week' and 'Hour'
#standard_hours_last_4_weeks['Week-Hour'] = standard_hours_last_4_weeks.apply(lambda row: f"Week {row['Week']}-Hour {row['Hour']}", axis=1)

# Group by 'Week' and 'Hour' and count the occurrences of each combination
#hour_counts = standard_hours_last_4_weeks.groupby(['Week', 'Hour']).size().reset_index(name='Count')

# Create a new column combining 'Week' and 'Hour' for plotting
#hour_counts['Week-Hour'] = hour_counts.apply(lambda row: f"Week {row['Week']}-Hour {row['Hour']}", axis=1)

# Plot the data using Plotly Express
#fig = px.bar(
#    hour_counts, 
#    x='Week-Hour', 
#    y='Count', 
#    color='Week', 
#    title=f'COUNT OF INSTANCES HAVING {KPI_TO_ANALYZE} > 0<br>WEEK WISE FOR LAST 4 WEEKS',
#    labels={'Week-Hour': 'Week-Hour', 'Count': 'Count'}
#)

# Adjust the x-axis and y-axis titles
#fig.update_xaxes(title='Week-Hour')
#fig.update_yaxes(title='Count of Hours')

# Adjust the size of the chart
#fig.update_layout(
#    width=1000,  # Set the width of the chart
#    height=500,  # Set the height of the chart
#)

#fig.show()


# In[33]:


# Filter the data where KPI_TO_ANALYZE > 0
standard_hours = filtered_df[filtered_df[KPI_TO_ANALYZE] > 10]

# Get the last 4 weeks
standard_hours['Week'] = standard_hours['Date'].dt.isocalendar().week
last_4_weeks = sorted(standard_hours['Week'].unique())[-4:]

# Filter data for the last 4 weeks
standard_hours_last_4_weeks = standard_hours[standard_hours['Week'].isin(last_4_weeks)]

# Create a new column combining 'Week' and 'Hour'
standard_hours_last_4_weeks['Week-Hour'] = standard_hours_last_4_weeks.apply(lambda row: f"Week {row['Week']}-Hour {row['Hour']}", axis=1)

# Group by 'Week' and 'Hour' and count the occurrences of each combination
hour_counts = standard_hours_last_4_weeks.groupby(['Week', 'Hour']).size().reset_index(name='Count')

# Create a new column combining 'Week' and 'Hour' for plotting
hour_counts['Week-Hour'] = hour_counts.apply(lambda row: f"Week {row['Week']}-Hour {row['Hour']}", axis=1)

# Plot the data using Plotly Express
fig = px.bar(
    hour_counts, 
    x='Week-Hour', 
    y='Count', 
    color='Week', 
    title=f'COUNT OF INSTANCES HAVING {KPI_TO_ANALYZE} > 0<br>WEEK WISE FOR LAST 4 WEEKS',
    labels={'Week-Hour': 'Week-Hour', 'Count': 'Count'}
)

# Add shaded area with 50% transparency from 6pm till 11pm for each week
for week in last_4_weeks:
    fig.add_shape(
        type="rect",
        xref="x",
        yref="paper",
        x0=f"Week {week}-Hour 19",
        y0=0,
        x1=f"Week {week}-Hour 23",
        y1=1,
        fillcolor="rgba(0,0,255,0.2)",
        layer="below",
        line_width=0,
    )

# Adjust the x-axis and y-axis titles
fig.update_xaxes(title='Week-Hour')
fig.update_yaxes(title='Count of Hours')

# Adjust the size of the chart
fig.update_layout(
    width=1000,  # Set the width of the chart
    height=500,  # Set the height of the chart
)

fig.show()


# #### DAY WISE OPERATING HOURS

# In[34]:


#---------
# Daywise
#---------

# Filter the dataframe where KPI_TO_ANALYZE > 0
filtered_operatinghours_day = filtered_df[filtered_df[KPI_TO_ANALYZE] > 10]

# Group by 'Date' and count the number of hours for each date
grouped_df = filtered_operatinghours_day.groupby('Date').size().reset_index(name='Count_of_Hours')

# Calculate the average count
average_count = grouped_df['Count_of_Hours'].mean()

# Plot the result using Plotly Express
fig = px.bar(grouped_df, x='Date', y='Count_of_Hours', title=f'COUNT OF INSTANCES HAVING {KPI_TO_ANALYZE} > 0 <br>DAY WISE TREND')

# Resize the chart
fig.update_layout(
    width=1000,  # Set the width of the chart
    height=500  # Set the height of the chart
)

# Add a threshold line at the average value
fig.add_shape(
    type="line",
    x0=grouped_df['Date'].min(),
    y0=average_count,
    x1=grouped_df['Date'].max(),
    y1=average_count,
    line=dict(
        color="Red",
        width=2,
        dash="dashdot",
    ),
    name="Average"
)

# Show the plot
fig.show()


# #### WEEK WISE OPERATING HOURS

# In[35]:


#-----------
# WEEK WISE
#-----------

# Filter the dataframe where KPI_TO_ANALYZE > 0
filtered_operatinghours_week = filtered_df[filtered_df[KPI_TO_ANALYZE] > 10]

# Convert the 'Date' column to a week period
filtered_operatinghours_week['Week'] = pd.to_datetime(filtered_operatinghours_week['Date']).dt.isocalendar().week

# Group by 'Week' and count the number of hours for each week
grouped_df = filtered_operatinghours_week.groupby('Week').size().reset_index(name='Count_of_Hours')

# Calculate the average count
average_count = grouped_df['Count_of_Hours'].mean()

# Plot the result using Plotly Express
fig = px.bar(grouped_df, x='Week', y='Count_of_Hours', title=f'COUNT OF INSTANCES HAVING {KPI_TO_ANALYZE} > 0 <br>WEEK WISE TREND')

# Resize the chart
fig.update_layout(
    width=1000,  # Set the width of the chart
    height=500  # Set the height of the chart
)

# Add a threshold line at the average value
fig.add_shape(
    type="line",
    x0=grouped_df['Week'].min(),
    y0=average_count,
    x1=grouped_df['Week'].max(),
    y1=average_count,
    line=dict(
        color="Red",
        width=2,
        dash="dashdot",
    ),
    name="Average"
)

# Show the plot
fig.show()


# #### MONTH WISE OPERATING HOURS

# In[36]:


#-----------
# MONTH WISE
#-----------

# Filter the dataframe where KPI_TO_ANALYZE > 0
filtered_operatinghours_month = filtered_df[filtered_df[KPI_TO_ANALYZE] > 10]

# Convert 'Date' to datetime if not already in datetime format
filtered_operatinghours_month['Date'] = pd.to_datetime(filtered_operatinghours_month['Date'])

# Group by month and count the number of hours for each month
grouped_df = filtered_operatinghours_month.groupby(pd.Grouper(key='Date', freq='M')).size().reset_index(name='Count_of_Hours')

# Add month name or month number to the 'Date' column
grouped_df['Month'] = grouped_df['Date'].dt.strftime('%b')  # To show month name
# grouped_df['Month'] = grouped_df['Date'].dt.month  # To show month number

# Calculate the average count
average_count = grouped_df['Count_of_Hours'].mean()

# Plot the result using Plotly Express
fig = px.bar(grouped_df, x='Month', y='Count_of_Hours', title=f'COUNT OF INSTANCES HAVING {KPI_TO_ANALYZE} > 0 <br>MONTH WISE TREND')

# Resize the chart
fig.update_layout(
    width=1000,  # Set the width of the chart
    height=500  # Set the height of the chart
)

# Add a threshold line at the average value
fig.add_shape(
    type="line",
    x0=grouped_df['Month'].min(),
    y0=average_count,
    x1=grouped_df['Month'].max(),
    y1=average_count,
    line=dict(
        color="Red",
        width=2,
        dash="dashdot",
    ),
    name="Average"
)

# Show the plot
fig.show()


# ## COMPARING SAME TYPE OF LOAD

# In[37]:


# Ensure 'Date' column is in datetime format
filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])

# Melt the DataFrame to long format
melted_df = filtered_df.melt(id_vars=['Date', 'Hour'], value_vars=[KPI_TO_ANALYZE, 'U_2_ACTIVE_POWER_TOTAL_KW', 'U_23_ACTIVE_POWER_TOTAL_KW'], var_name='KPI', value_name='Value')

# Create a combined datetime column for x-axis
melted_df['Datetime'] = melted_df['Date'] + pd.to_timedelta(melted_df['Hour'], unit='h')

# Determine the KPIs to be plotted
line_kpi = 'U_2_ACTIVE_POWER_TOTAL_KW'
area_kpi = KPI_TO_ANALYZE
bar_kpi = 'U_23_ACTIVE_POWER_TOTAL_KW'

# Create the figure
fig = go.Figure()

# Add the line chart
fig.add_trace(go.Scatter(x=melted_df[melted_df['KPI'] == line_kpi]['Datetime'],
                        y=melted_df[melted_df['KPI'] == line_kpi]['Value'],
                        mode='lines',
                        name=line_kpi))

# Add the area chart
fig.add_trace(go.Scatter(x=melted_df[melted_df['KPI'] == area_kpi]['Datetime'],
                        y=melted_df[melted_df['KPI'] == area_kpi]['Value'],
                        mode='lines',
                        fill='tozeroy',
                        opacity=0.5,
                        name=area_kpi))

# Add the bar chart
fig.add_trace(go.Bar(x=melted_df[melted_df['KPI'] == bar_kpi]['Datetime'],
                    y=melted_df[melted_df['KPI'] == bar_kpi]['Value'],
                    opacity=0.5,
                    name=bar_kpi))

# Adjust the layout
fig.update_layout(
    width=1000,
    height=500,
    title='TRENDS FOR SAME TYPE OF LOAD OVER TIME',
    #xaxis_title='Date and Hour',
    yaxis_title='KPI Value',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.2,
        xanchor="center",
        x=0.5
    ),
    xaxis_type='date'
)

fig.show()


# In[38]:


# Select the columns you want to include in the new DataFrame
power_columns = filtered_df[['U_2_ACTIVE_POWER_TOTAL_KW', 'U_15_ACTIVE_POWER_TOTAL_KW', 'U_23_ACTIVE_POWER_TOTAL_KW']]

# Add an index column if it doesn't already exist
power_columns['Index'] = power_columns.index

# Create the 3D scatter plot
fig_meter = px.scatter_3d(
    power_columns, 
    x='U_2_ACTIVE_POWER_TOTAL_KW', 
    y='U_15_ACTIVE_POWER_TOTAL_KW', 
    z='U_23_ACTIVE_POWER_TOTAL_KW',
    color='Index',  # Color by meter index
    title='3D Scatter Plot of Meters',
    labels={
        'U_2_ACTIVE_POWER_TOTAL_KW': 'U2 Power', 
        'U_15_ACTIVE_POWER_TOTAL_KW': 'U15 Power', 
        'U_23_ACTIVE_POWER_TOTAL_KW': 'U23 Power'
    }
)

# Update layout dimensions
fig_meter.update_layout(width=1000, height=600)

# Show the plot
fig_meter.show()


# ## THRESHOLDS

# We can also check Count of Hours exceeding threshold. Also you can check Delta in power factors, Delta in phase currents on similar lines

# In[39]:


# Ensure 'Date' column is in datetime format
filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])

# Melt the DataFrame to long format
melted_df = filtered_df.melt(id_vars=['Date', 'Hour'], value_vars=['U_15_HARMONICS_I1_THD', 'U_15_HARMONICS_I2_THD', 'U_15_HARMONICS_I3_THD'], var_name='KPI', value_name='Value')

# Create a combined datetime column for x-axis
melted_df['Datetime'] = melted_df['Date'] + pd.to_timedelta(melted_df['Hour'], unit='h')

# Plot the data using Plotly Express
fig = px.line(melted_df, x='Datetime', y='Value', color='KPI', title='HARMONICS TREND OVER TIME <br>AGAINST THRESHOLD', labels={'Value': 'KPI Value', 'Datetime': 'Date and Hour'})

# Add a threshold line at 50
fig.add_shape(
    type="line",
    x0=melted_df['Datetime'].min(), y0=50,
    x1=melted_df['Datetime'].max(), y1=50,
    line=dict(color="Red", width=2, dash="dash"),
)

# Adjust the size of the chart and move the legend to the bottom
fig.update_layout(
    width=1000,  # Set the width of the chart
    height=500,  # Set the height of the chart
    legend=dict(
        orientation="h",  # Horizontal orientation
        yanchor="bottom",  # Anchor the legend at the bottom
        y=-0.2,  # Adjust this value to place the legend inside or outside the plot area
        xanchor="center",  # Center the legend horizontally
        x=0.5  # Position the legend at the center of the x-axis
    )
)
fig.show()


# ## PERCENTAGE SHARES

# In[40]:


# Calculate the total for each KPI
kpi_totals = filtered_df[['U_2_ACTIVE_POWER_TOTAL_KW', 'U_3_ACTIVE_POWER_TOTAL_KW', 'U_4_ACTIVE_POWER_TOTAL_KW','U_5_ACTIVE_POWER_TOTAL_KW','U_6_ACTIVE_POWER_TOTAL_KW','U_7_ACTIVE_POWER_TOTAL_KW','U_8_ACTIVE_POWER_TOTAL_KW','U_9_ACTIVE_POWER_TOTAL_KW','U_10_ACTIVE_POWER_TOTAL_KW','U_11_ACTIVE_POWER_TOTAL_KW','U_12_ACTIVE_POWER_TOTAL_KW','U_13_ACTIVE_POWER_TOTAL_KW','U_14_ACTIVE_POWER_TOTAL_KW','U_15_ACTIVE_POWER_TOTAL_KW','U_16_ACTIVE_POWER_TOTAL_KW','U_17_ACTIVE_POWER_TOTAL_KW','U_18_ACTIVE_POWER_TOTAL_KW','U_19_ACTIVE_POWER_TOTAL_KW','U_20_ACTIVE_POWER_TOTAL_KW','U_21_ACTIVE_POWER_TOTAL_KW','U_22_ACTIVE_POWER_TOTAL_KW']].sum().reset_index()
kpi_totals.columns = ['KPI', 'Total']

# Calculate the percentage of power for each KPI
kpi_totals['Percentage'] = (kpi_totals['Total'] / kpi_totals['Total'].sum()) * 100

# Sort the table by percentage in descending order
kpi_totals = kpi_totals.sort_values(by='Percentage', ascending=False)

# Select the top 15 KPIs
top_15_kpis = kpi_totals.head(15)

# Define the colors for the pie chart
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1a55FF', '#00CC00', '#FF00FF', '#00FFFF', '#FFFF00']

# Create the pie chart
fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'pie'}, {'type': 'table'}]], 
                   subplot_titles=("PERCENTAGE OF POWER DRAWN", "TOP-N POWER CONSUMPTION PERCENTAGES"))

# Add the pie chart
fig.add_trace(go.Pie(labels=top_15_kpis['KPI'], values=top_15_kpis['Total'], hole=0.6, 
                    textinfo='percent', textfont_size=12, 
                    marker=dict(colors=colors[:len(top_15_kpis)])), row=1, col=1)

# Add the data table
fig.add_trace(go.Table(header=dict(values=['KPI', 'Percentage'],
                                 fill_color='paleturquoise',
                                 align='left'),
                      cells=dict(values=[top_15_kpis['KPI'], top_15_kpis['Percentage'].round(1)],
                                fill_color='lavender',
                                align='left')), row=1, col=2)

# Customize the layout
fig.update_layout(title='POWER CONSUMPTION ANALYSIS',
                  title_x=0.5,
                  width=1000, 
                  height=500,
                  bargap=0.1)

fig.show()


# 
# ## SEASONAL DECOMPOSITION

# #### CREATING COPY OF FILTERED_DF

# In[41]:


filtered_df.head()
filtered_df_copy = filtered_df.copy()


# In[42]:


# Convert 'PLC_DATE_TIME' column to datetime
filtered_df['PLC_DATE_TIME'] = pd.to_datetime(filtered_df['PLC_DATE_TIME'])

# Convert 'KPI_TO_ANALYZE' column to numeric if it's not already
filtered_df[KPI_TO_ANALYZE] = pd.to_numeric(filtered_df[KPI_TO_ANALYZE], errors='coerce')

# Drop rows with NaN values if any
filtered_df.dropna(subset=[KPI_TO_ANALYZE], inplace=True)

# Set 'PLC_DATE_TIME' as index
filtered_df.set_index('PLC_DATE_TIME', inplace=True)

# Perform seasonal decomposition
decomposition = seasonal_decompose(filtered_df[KPI_TO_ANALYZE], model='additive', period=24)  # Assuming daily seasonality

# Plot the decomposition
fig, axes = plt.subplots(4, 1, figsize=(12, 10))
decomposition.observed.plot(ax=axes[0], title='Observed')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
decomposition.resid.plot(ax=axes[3], title='Residuals')
plt.tight_layout()
plt.show()


# In[43]:


#filtered_df.head(4)


# In[44]:


# Convert 'PLC_DATE_TIME' column to datetime
filtered_df_copy[DATE_COL] = pd.to_datetime(filtered_df_copy[DATE_COL])

# Convert 'KPI_TO_ANALYZE' column to numeric if it's not already
filtered_df_copy[KPI_TO_ANALYZE] = pd.to_numeric(filtered_df_copy[KPI_TO_ANALYZE], errors='coerce')

# Drop rows with NaN values if any
filtered_df_copy.dropna(subset=[KPI_TO_ANALYZE], inplace=True)

# Set 'PLC_DATE_TIME' as index
#filtered_df_copy.set_index(DATE_COL, inplace=True)

# Perform seasonal decomposition
decomposition = seasonal_decompose(filtered_df_copy[KPI_TO_ANALYZE], model='additive', period=24)  # Assuming daily seasonality

# Create a figure with subplots using Plotly Express
fig = go.Figure()

# Add observed, trend, seasonal, and residual plots as subplots
fig.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, mode='lines', name='Observed'))
fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend'))
fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonal'))
fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='lines', name='Residuals'))

# Update layout and display the figure
fig.update_layout(title='Seasonal Decomposition of Time Series',
                  xaxis_title='Date',
                  yaxis_title='Value',
                  height=500,
                  width=1000,
                  template='plotly_white')

fig.show()


# ## ALERTER

# In[45]:


# Extract date and hour
filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])

# Filter the last 10 available days
latest_date = filtered_df['Date'].max()
last_10_days = filtered_df['Date'] >= (latest_date - pd.Timedelta(days=9))
filtered_df = filtered_df.loc[last_10_days]

# Create the cross-tab
cross_tab = filtered_df.pivot_table(index='Hour', columns='Date', values=KPI_TO_ANALYZE, aggfunc='mean')

# Ensure all 10 days are included in the cross-tab
all_dates = pd.date_range(start=latest_date - pd.Timedelta(days=9), end=latest_date, freq='D')
cross_tab = cross_tab.reindex(columns=all_dates)

# Create the heatmap with values
fig = px.imshow(cross_tab, color_continuous_scale='Viridis', title=f'{KPI_TO_ANALYZE} BY HOUR & DATE<br>LAST 10 DAYS')
fig.update_traces(text=cross_tab.values, texttemplate='%{text:.2f}')
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Hour',
    font_size=14,
    width=900,
    height=800,
    xaxis_tickformat='%b %d',  # Use a more compact date format
    xaxis_tickangle=-45,  # Rotate the date labels for better visibility
    # Set fixed tick values (replace with your actual dates if needed)
    xaxis_tickvals=all_dates.tolist()
)
fig.show()

