"""
DHV Assignment: Infographics Project
Git hub link: https://github.com/riyazchoorikhanrk/DHV-Assignment
Student number: 22096816
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.read_csv(
    "gdp.csv")  # gdp
df2 = pd.read_csv(
    "gdp_growth.csv")  # gdp_growth

df1.head()

df2.head()

df1.columns

df1.shape

"""# Exploratory Data ANalysis"""

df1.isnull().sum()

"""## Cleaning the data"""

# Data Cleaning
# Example: Handling Missing Values
df1.fillna(0, inplace=True)  # Replace NaN values with 0, you might choose a
# different strategy
df2.fillna(0, inplace=True)

df1.isnull().sum()
df2.isnull().sum()

# Deleting the 'Unnamed: 65' column

df1 = df1.drop("Unnamed: 65", axis=1)
df2 = df2.drop("Unnamed: 65", axis=1)

# Rename the 'OldColumnName' to 'NewColumnName'
df1 = df1.rename(columns={'Country Name': 'Country_Name'})
df2 = df2.rename(columns={'Country Name': 'Country_Name'})

df1.shape

"""## Summary statistics using pandas"""

df1.describe()

"""## Summary statistics using Numpy"""

gdp_values_2020 = df1['2020']

summary_statistics_numpy_2020 = {
    'mean': np.mean(gdp_values_2020),
    'median': np.median(gdp_values_2020),
    'std_dev': np.std(gdp_values_2020),
    'min': np.min(gdp_values_2020),
    'max': np.max(gdp_values_2020)
}

summary_statistics_numpy_2020

"""## Using  matplotlib and seaborn libraries to produce visualisations for the above data"""

rows_to_remove = ["Middle income", "North America", "Late-demographic dividend",
                  "Europe & Central Asia", "East Asia & Pacific (excluding high income)", "East Asia & Pacific (IDA & IBRD countries)",
                  "European Union", "Euro area",	"World", "High income", "OECD members",
                  "Post-demographic dividend",
                  "IDA & IBRD total", "Low & middle income", "Middle income	", "IBRD only",
                  "East Asia & Pacific", "Upper middle income"
                  "Early-demographic dividend ", "Lower middle income",
                  "Latin America & Caribbean",
                  "Latin America & the Caribbean (IDA & IBRD countries)",
                  "Latin America & Caribbean (excluding high income)",
                  "Europe & Central Asia (IDA & IBRD countries)",
                  "Early-demographic dividend", "South Asia ", "South Asia (IDA & IBRD)", "Europe & Central Asia (excluding high income)",
                  "Middle East & North Africa" "Early-demographic dividend",
                  "South Asia	", "Arab World", "South Asia", "IDA total", "Sub-Saharan Africa",
                  "Sub-Saharan Africa (IDA & IBRD countries)	",
                  "Sub-Saharan Africa (IDA & IBRD countries)",
                  "Sub-Saharan Africa (excluding high income)	",
                  "Sub-Saharan Africa (excluding high income)"
                  "Central Europe and the Baltics",
                  "Sub-Saharan Africa (excluding high income)",
                  "Central Europe and the Baltics	", "Central Europe and the Baltics",
                  "Upper middle income", "Middle East & North Africa	", "Middle East & North Africa", "Aruba", "Andorra", "	Channel Islands	", "Eritrea",
                  "Faroe Islands", "Gibraltar", "Greenland", "Isle of Man",
                  "Not classified	", "Liechtenstein",
                  "Channel Islands", "Sint Maarten (Dutch part)	", "South Sudan	", "San Marino", "French Polynesia	", "Korea, Dem. People's Rep.",
                  "New Caledonia", "Northern Mariana Islands	",
                  "St. Martin (French part)", "Not classified", 'Northern Mariana Islands',
                  'French Polynesia',
                  'South Sudan',
                  'Sint Maarten (Dutch part)',
                  'Syrian Arab Republic',
                  'Turkmenistan',
                  'Venezuela, RB',
                  'British Virgin Islands',
                  'Virgin Islands (U.S.)',
                  'Yemen, Rep.']

df1 = df1[~df1['Country_Name'].isin(rows_to_remove)]
df2 = df2[~df2['Country_Name'].isin(rows_to_remove)]


def plot_gdp_comparison(df):
    """
    Plots a comparison of GDP for the top and bottom 10 countries in a specified year.

    Parameters:
        df (DataFrame): The DataFrame containing GDP data.

    Returns:
        None. The function saves the plot as an image file.
    """
    gdp_columns = df.columns[2:]
    latest_year = '2020'

    top_countries_latest_year = df[[
        'Country_Name', latest_year]].nlargest(10, latest_year)
    bottom_countries_latest_year = df[[
        'Country_Name', latest_year]].nsmallest(10, latest_year)

    top_countries_latest_year[latest_year] = pd.to_numeric(
        top_countries_latest_year[latest_year])
    bottom_countries_latest_year[latest_year] = pd.to_numeric(
        bottom_countries_latest_year[latest_year])

    plt.figure(figsize=(15, 8))

    plt.subplot(1, 2, 1)
    top_chart = sns.barplot(x=top_countries_latest_year[latest_year] / 1e12,
                            y='Country_Name', data=top_countries_latest_year, palette='viridis')
    plt.title(f'Top 10 Countries by GDP in {latest_year}')
    plt.xlabel('GDP (in Trillions)')
    plt.ylabel('Country')

    for p in top_chart.patches:
        top_chart.annotate(f'{p.get_width():.2f} Trillion', (p.get_width(),
                                                             p.get_y() + p.get_height() / 2), ha='center', va='center',
                           xytext=(5, 0), textcoords='offset points')

    plt.subplot(1, 2, 2)
    bottom_chart = sns.barplot(x=bottom_countries_latest_year[latest_year] / 1e12,
                               y='Country_Name', data=bottom_countries_latest_year, palette='viridis')
    plt.title(f'Bottom 10 Countries by GDP in {latest_year}')
    plt.xlabel('GDP (in Trillions)')
    plt.ylabel('Country')

    for p in bottom_chart.patches:
        bottom_chart.annotate(f'{p.get_width():.5f} Trillion', (p.get_width(),
                                                                p.get_y() + p.get_height() / 2), ha='center', va='center',
                              xytext=(5, 0), textcoords='offset points')

    plt.subplots_adjust(wspace=0.3)
    plt.tight_layout()
    plt.savefig("22096816.png", dpi=300)


plot_gdp_comparison(df1)


def plot_gdp_growth(df):
    """
    Plots the GDP growth over 10-year intervals for the top 5 performing countries in 2020.

    Parameters:
        df (DataFrame): The DataFrame containing GDP data.

    Returns:
        None. The function saves the plot as an image file.
    """
    gdp_columns = df.columns[2:]
    top_countries_2020 = df[['Country_Name', '2020']].nlargest(10, '2020')

    global_gdp_growth_top_countries = df[df['Country_Name'].isin(
        top_countries_2020['Country_Name'])][gdp_columns].T

    global_gdp_growth_top_countries.index = pd.to_datetime(
        global_gdp_growth_top_countries.index)

    global_gdp_growth_10_years = global_gdp_growth_top_countries.resample(
        '10A').mean()

    global_gdp_growth_10_years.columns = df['Country_Name'][global_gdp_growth_10_years.columns].tolist(
    )

    plt.figure(figsize=(12, 6))

    for country in global_gdp_growth_10_years.columns:
        sns.lineplot(x=global_gdp_growth_10_years.index,
                     y=global_gdp_growth_10_years[country], label=country)

    plt.legend(title='Country', loc='upper left', bbox_to_anchor=(1, 1))

    plt.title(
        'Performance of Top 10  Countries GDP Growth Over 10-Year Intervals from (1960-2020)')
    plt.xlabel('Year Interval')
    plt.ylabel('GDP Growth (Average)')
    plt.grid(True)
    plt.savefig("22096816_1.png", dpi=300)


plot_gdp_growth(df1)


def plot_china_gdp_growth_comparison(df):
    """
    Plots the annual GDP growth comparison of China for the years 2005 and 2019.

    Parameters:
        df (DataFrame): The DataFrame containing GDP data.

    Returns:
        None. The function saves the plot as an image file.
    """
    gdp_growth_2005 = df[['Country_Name'] + ['2005']]
    gdp_growth_2019 = df[['Country_Name'] + ['2019']]

    top_5_countries_2019 = gdp_growth_2019.nlargest(5, '2019', 'all')
    top_5_countries_2005 = gdp_growth_2005.nlargest(5, '2005', 'all')

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 2)
    colors_2005 = sns.color_palette('pastel')
    colors_2005[4] = 'orange'
    explode_2005 = (0, 0, 0, 0, 0.2)
    plt.pie(top_5_countries_2005['2005'], labels=top_5_countries_2005['Country_Name'],
            autopct='%1.1f%%', startangle=90, colors=colors_2005, explode=explode_2005)
    plt.title('Top 5 Performing Countries in 2005')

    plt.subplot(1, 2, 1)
    colors_2019 = sns.color_palette('pastel')
    colors_2019[1] = 'orange'
    explode_2019 = (0, 0.2, 0, 0, 0)
    plt.pie(top_5_countries_2019['2019'], labels=top_5_countries_2019['Country_Name'],
            autopct='%1.1f%%', startangle=90, colors=colors_2019, explode=explode_2019)
    plt.title('Top 5 Performing Countries in 2019')

    plt.suptitle('Annual GDP Growth Comparison of China (2005 vs 2019)')
    plt.savefig("22096816_2.png", dpi=300)


plot_china_gdp_growth_comparison(df1)


def plot_gdp_growth_comparison(df):
    """
    Plots the GDP growth comparison for specified countries between the years 2019 and 2020.

    Parameters:
        df (DataFrame): The DataFrame containing GDP data with 'Country_Name' as the index.

    Returns:
        None. The function displays the plot using matplotlib and seaborn.
    """
    df.set_index('Country_Name', inplace=True)

    countries_of_interest = ["United States",
                             "China", "Japan", "Germany", "United Kingdom"]
    df_countries = df.loc[countries_of_interest, ['2019', '2020']]

    df_countries = df_countries.T

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_countries, markers=False)
    plt.title('GDP Growth Comparison (2019 vs. 2020) in Covid-19 Pandemic')
    plt.xlabel('Year')
    plt.ylabel('GDP Value')
    plt.legend(title='Country')
    plt.savefig("22096816.png", dpi=300)


plot_gdp_growth_comparison(df2)
