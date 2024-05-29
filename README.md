# movie-reviews
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LogisticRegression

%matplotlib inline

import warnings
warnings.simplefilter('ignore')

# Install necessary libraries
pip install datasets
pip install transformers
pip install sklearn

from datasets import load_dataset

# Load the IMDb dataset
dataset = load_dataset("imdb")


import os

df = pd.read_csv(r"C:\Users\Anjali Kumari\Downloads\archive (7)\IMDB-Movie-Data.csv")

df.head(5)

df.tail(10)

df.shape

print("No. of Rows: ",df.shape[0])
print("No. of Columns: ",df.shape[1])

df.info()

df.isnull().values.any()

df.isnull().sum()

sns.heatmap(df.isnull())

per_missing_values = df.isnull().sum()*100/len(df)

per_missing_values

df.dropna()

df.duplicated().any()

df.describe(include = 'all')

df.columns

df[df['Runtime (Minutes)'] >= 180]['Title']

df.groupby('Year')['Votes'].mean().sort_values(ascending=False)

sns.barplot(x='Year', y='Votes', data=df)
plt.title('Votes by Year')
plt.show()


df.groupby('Year')['Revenue (Millions)'].mean().sort_values(ascending=False)

sns.barplot(x='Year', y='Revenue (Millions)', data=df)
plt.title('Revenues by Year')
plt.show()

df.groupby('Director')['Rating'].mean().sort_values(ascending=False)

top_ten = df.nlargest(10, 'Runtime (Minutes)')[['Title','Runtime (Minutes)']].set_index('Title')

top_ten

sns.barplot(x='Runtime (Minutes)', y=top_ten.index, data=top_ten)
plt.show()

df.groupby("Year")['Title'].count().sort_values(ascending=False)

df['Year'].value_counts()

sns.countplot(x='Year', data=df)
plt.title('No. Of Movies Per Year')
plt.show()

df[df['Revenue (Millions)'].max() == df['Revenue (Millions)']]['Title']

top10 = df.nlargest(10, 'Rating')[['Title', 'Director', 'Rating']].set_index('Title')

top10

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming top10 is a DataFrame containing the top 10 movies and their ratings and directors
# Ensure 'Rating' and 'Director' columns exist in the DataFrame

# Plot the data
sns.barplot(x='Rating', y=top10.index, data=top10, hue='Director', dodge=False)
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.title('Top 10 Movies Ratings by Director')
plt.xlabel('Rating')
plt.ylabel('Movie Title')
plt.show()


top10 = df.nlargest(10, 'Revenue (Millions)')[['Title', 'Revenue (Millions)']].set_index('Title')

top10

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming top10 is a DataFrame containing the top 10 movies and their revenue
# Ensure 'Revenue (Millions)' column exists in the DataFrame

# Plot the data
sns.barplot(x='Revenue (Millions)', y=top10.index, data=top10)
plt.title('Top 10 Highest Revenue Movie Titles')
plt.xlabel('Revenue (Millions)')
plt.ylabel('Movie Title')
plt.show()


df.groupby('Year')['Rating'].mean()

sns.scatterplot(x='Rating', y='Revenue (Millions)', data = df)

def rating(rating):
    if rating >= 7.0:
        return 'Excellent'
    elif rating >=6.0:
        return 'Good'
    else: return 'Average'

df['Rating_Cateory'] = df['Rating'].apply(rating)

df.head()

len(df[df['Genre'].str.contains('action', case=False)])

list1 = []
for value in df['Genre']:
    list1.append(value.split(','))

list1

one_d = []
for item in list1:
    for item1 in item:
        one_d.append(item1)

uni_list = []
for item in one_d:
    if item not in uni_list:
        uni_list.append(item)

uni_list

one_d = []
for item in list1:
    for item1 in item:
        one_d.append(item1)

from collections import Counter

Counter(one_d)

       
