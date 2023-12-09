#!/usr/bin/env python
# coding: utf-8

# ## EDA on Cricket World Cup Data

# #### importing libraries

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"
from scipy import stats


# In[2]:


# Set display options to show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[3]:


# Load the datasets
innings_df = pd.read_csv(r'C:\Users\ibrahim laptops\Documents\Datasets\Cricket_dataset\ICC Cricket World Cup\innings.csv')
matches_df = pd.read_csv(r'C:\Users\ibrahim laptops\Documents\Datasets\Cricket_dataset\ICC Cricket World Cup\matches.csv')
overBallDetails_df = pd.read_csv(r'C:\Users\ibrahim laptops\Documents\Datasets\Cricket_dataset\ICC Cricket World Cup\overBallDetails.csv')
overHistory_df = pd.read_csv(r'C:\Users\ibrahim laptops\Documents\Datasets\Cricket_dataset\ICC Cricket World Cup\overHistory.csv')
players_df = pd.read_csv(r'C:\Users\ibrahim laptops\Documents\Datasets\Cricket_dataset\ICC Cricket World Cup\players.csv')
teams_df = pd.read_csv(r'C:\Users\ibrahim laptops\Documents\Datasets\Cricket_dataset\ICC Cricket World Cup\teams.csv')
venues_df = pd.read_csv(r'C:\Users\ibrahim laptops\Documents\Datasets\Cricket_dataset\ICC Cricket World Cup\venues.csv')


# In[4]:


# Replace 'matches_df' with the actual name of your DataFrame
matches_df = matches_df.drop(matches_df.columns[25:174], axis=1)


# In[5]:


common_features = set(innings_df.columns)

for df in [matches_df, overBallDetails_df, overHistory_df, players_df, teams_df, venues_df]:
    common_features = common_features.intersection(df.columns)

# Now common_features contains the set of features that are present in all 7 dataframes
print(common_features)


# In[6]:


# Rename columns to avoid conflicts during merging
innings_df = innings_df.rename(columns={'id': 'innings_id'})
matches_df = matches_df.rename(columns={'id': 'match_id'})
overBallDetails_df = overBallDetails_df.rename(columns={'id': 'over_ball_id'})
overHistory_df = overHistory_df.rename(columns={'id': 'over_history_id'})
players_df = players_df.rename(columns={'id': 'player_id'})
teams_df = teams_df.rename(columns={'id': 'team_id'})
venues_df = venues_df.rename(columns={'id': 'venue_id'})


# In[7]:


# Merge DataFrames based on common columns
df = pd.merge(innings_df, matches_df, how='inner', left_on='innings_id', right_on='match_id')
df = pd.merge(df, overBallDetails_df, how='inner', left_on='innings_id', right_on='over_ball_id')
df = pd.merge(df, overHistory_df, how='inner', left_on='overHistoryId', right_on='over_history_id')
df = pd.merge(df, players_df, how='inner', left_on='facingBatsmanId', right_on='player_id')
df = pd.merge(df, teams_df, how='inner', left_on='teamId', right_on='team_id')
df = pd.merge(df, venues_df, how='inner', left_on='venueId', right_on='venue_id')


# In[8]:


df.head()


# ### Data Cleaning and Preprocessing

# In[9]:


df.dtypes


# In[10]:


df.info()


# In[11]:


df.columns


# In[12]:


df.duplicated().sum()


# In[13]:


df.isnull().sum()


# In[14]:


# Assuming matches_df is your dataframe
matches_df['matchDate'] = pd.to_datetime(matches_df['matchDate'], errors='coerce', utc=True)
matches_df['matchEndDate'] = pd.to_datetime(matches_df['matchEndDate'], errors='coerce', utc=True)


# In[16]:


df.describe()


# Droping the irrelevant columns form the dataset.
# 
# 
# Columns were dropped based on their irrelevance to the specified analysis tasks, high missing values, and the presence of redundant or detailed information. This streamlined the dataset, focusing on essential metrics and trends for a more meaningful Exploratory Data Analysis (EDA) of the Cricket World Cup.

# In[17]:


# List of columns to drop
columns_to_drop = [
    'overProgress', 'byeRuns', 'legByeRuns', 'penaltyRuns',
    'countingBall', 'nonCountingBall', 'over_ball_id',
    'matchStatus_victoryMarginRuns', 'matchStatus_victoryMarginWickets', 'matchStatus_victoryMarginInningsRuns',
    'matchDateMs', 'matchEndDateMs',
    'ovBalls/0', 'ovBalls/1', 'ovBalls/2', 'ovBalls/3', 'ovBalls/4', 'ovBalls/5',
    'ovBalls/6', 'ovBalls/7', 'ovBalls/8', 'ovBalls/9', 'ovBalls/10', 'ovBalls/11',
    'ovBalls/12', 'ovBalls/13'
]

df1 = df.drop(columns=columns_to_drop, inplace=False)

df1.head()


# In[18]:


df1 = df1.drop(['matchSummary', 'totalBalls', 'isLimitedOvers', 'match.summary', 'matchStatus', 'umpire.name.5'], axis=1)


# In[19]:


df1.isnull().sum()


# In[20]:


df1.columns


# In[21]:


default_bowling_style = 'Default_Bowling_Style'
df1['bowlingStyle'].fillna(default_bowling_style, inplace=True)


# In[22]:


# Impute missing values in 'battingTeamId' column with the mode
batting_team_mode = df1['battingTeamId'].mode()[0]
df1['battingTeamId'].fillna(batting_team_mode, inplace=True)

# Impute missing values in 'bowlingTeamId' column with the mode
bowling_team_mode = df1['bowlingTeamId'].mode()[0]
df1['bowlingTeamId'].fillna(bowling_team_mode, inplace=True)


# Now I have removed missing values as much as I can in the dataset and also drop irrelevant column which we are not using for our analyis.

# ### Ouliters Detection

# In[23]:


# Select only the numeric columns for calculating Z-scores
numeric_columns = df1.select_dtypes(include='number').columns
df1_numeric = df1[numeric_columns]

# Calculate Z-scores for each data point
z_scores = stats.zscore(df1_numeric)

# Define a threshold for identifying outliers (e.g., 3 standard deviations)
threshold = 3

# Identify outliers
outliers = (z_scores > threshold).any(axis=1)

# Display rows with outliers
outlier_rows = df1[outliers]
outlier_rows.head()


# In[24]:


df1['matchDate']


# In[25]:


# Visualize the DataFrame using box plots
for column in numeric_columns:
    sns.boxplot(df1[column])
    plt.title(f'Box Plot of {column} (Outliers Removed)')
    plt.show()


# In[ ]:





# ### Exploratory Data Analysis

# I have devided analysis in differnt parts where I perform analysis on features of differnt dataframes to find trends on innings, matches, team evolution, venues, umpires, and player styles.

# What is the trend in the number of matches played over the years?

# In[26]:


# Convert 'matchDate' to datetime format
df['matchDate'] = pd.to_datetime(df['matchDate'], errors='coerce', utc=True)

# Group by year and count the number of matches
matches_each_year = df.groupby(df['matchDate'].dt.year)['matchId'].count().reset_index()

# Rename columns
matches_each_year.columns = ['Year', 'Matches']

# Visualize the trend in the number of matches played over the years
plt.plot(matches_each_year['Year'], matches_each_year['Matches'])


# Are there any patterns in the distribution of innings outcomes (runs, Innings, etc.)?

# In[29]:


# Histogram for Runs
plt.subplot(1, 2, 1)
plt.hist(df['runs'], bins=20, color='skyblue', edgecolor='black')

# Histogram for Wickets
plt.subplot(1, 2, 2)
plt.hist(df['inningsNumber'], bins=10, color='lightcoral', edgecolor='black')


# How have batting runs, balls faced, and strike rates changed over time, and what is the distribution of strike rates across different tournaments?

# In[35]:


# Line charts for batting averages, strike rates, and run rates over the years
sns.lineplot(x='matchDate', y='runs', data=df, label='Runs')
sns.lineplot(x='matchDate', y='ballsFaced', data=df, label='Balls Faced')
plt.title('Batting Stats Trends Over Time')
plt.xlabel('Match Date')
plt.ylabel('Count')
plt.legend()
plt.show()


# In[37]:


# Calculate strike rate and add it as a new column
df['strikeRate'] = (df['runs'] / df['ballsFaced']) * 100

# Box plot to show the distribution of batting performance metrics
sns.boxplot(x='tournamentLabel', y='strikeRate', data=df)
plt.title('Distribution of Strike Rates Across Tournaments')
plt.xlabel('Tournament')
plt.ylabel('Strike Rate')
plt.show()


# How have bowling wickets, economy rates, and performance in different match situations changed over time?

# In[39]:


# Heatmap to visualize the performance of bowlers in different match situations
heatmap_data = df.pivot_table(index='bowlerId_x', columns='matchStatus_outcome', values='wkts', aggfunc='count')
sns.heatmap(heatmap_data, cmap='Blues', annot=True, fmt='g')
plt.title('Bowler Performance Heatmap')
plt.xlabel('Match Outcome')
plt.ylabel('Bowler')
plt.show()


# What is the distribution of total scores in matches, and how do scores vary based on match outcomes?

# In[40]:


# Histogram to illustrate the distribution of total scores in matches
plt.hist(df['score'], bins=20, edgecolor='black')
plt.title('Distribution of Scores in Matches')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()


# How does the performance in terms of runs scored vary among different teams, and how can we visually represent team performance based on runs, wickets, and matches played?

# In[52]:


# Increase figure size to provide more space for x-axis labels
plt.figure(figsize=(12, 6))

# Rotate x-axis labels for better readability
sns.barplot(x='country', y='runs', data=df)
plt.xticks(rotation=45, ha='right')  # Adjust rotation angle and alignment as needed

plt.title('Runs Scored by Different Teams')
plt.xlabel('Country')
plt.ylabel('Runs')

plt.show()


# How do different venues impact match outcomes, and what is the distribution of matches across venues?

# In[59]:


# Bar chart to compare the impact of different venues on match outcomes
sns.barplot(x='venue_id', y='matchStatus_outcome', data=df)
plt.title('Impact of Venues on Match Outcomes')
plt.xlabel('Venue')
plt.ylabel('Mean Match Outcome')
plt.show()


# who are the top players with the most centuries?

# In[60]:


# Bar graph illustrating the number of centuries by players and their impact on match results
centuries_by_players = df['fullName_x'].value_counts().head(10)
centuries_by_players.plot(kind='bar')
plt.title('Top 10 Players with Most Centuries')
plt.xlabel('Player')
plt.ylabel('Number of Centuries')
plt.show()


# In[ ]:




