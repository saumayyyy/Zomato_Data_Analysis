import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
@st.cache_data
def load_data():
    # Load data
    zomato_data = pd.read_csv('zomato.csv', encoding='ISO-8859-1')
    country_data = pd.read_csv('Country-Code.csv')
    zomato_data = pd.merge(zomato_data, country_data, on='Country Code', how='left')
    
    # Data Cleaning
    zomato_data['Cuisines'] = zomato_data['Cuisines'].fillna("Unknown")  # Replace NaN in Cuisines
    zomato_data = zomato_data.dropna(subset=['Aggregate rating'])  # Drop rows with missing ratings
    
    
    return zomato_data.copy()

zomato_data = load_data()

# Currency conversion rates to USD (these are example rates, you should update them with real exchange rates)
currency_conversion = {
    'INR': 0.012,    
    'USD': 1.0,
    'EUR': 1.1,      
    'GBP': 1.3,     
    'AED': 0.27,    
    'SAR': 0.27,    
    'CAD': 0.75,    
    'ZAR': 0.06,    
    'CNY': 0.14,    
    'MYR': 0.24,    
}

# Convert cost to USD based on currency
def convert_cost(row):
    currency = row['Currency']  
    cost = row['Average Cost for two']
    return cost * currency_conversion.get(currency, 1)  # Default to 1 if currency is unknown

zomato_data['Converted Cost for Two (USD)'] = zomato_data.apply(convert_cost, axis=1)


st.sidebar.header("Filters")


country_options = ["All"] + sorted(zomato_data['Country'].unique())
selected_country = st.sidebar.selectbox("Select Country", options=country_options)

if selected_country == "All":
    total_restaurants = len(zomato_data)
else:
    total_restaurants = zomato_data[zomato_data['Country'] == selected_country].shape[0]

st.sidebar.markdown(f"**Total Restaurants: {total_restaurants}**")

# Rating filter
selected_rating = st.sidebar.slider(
    "Select Minimum Rating",
    min_value=0.0,
    max_value=5.0,
    value=3.0,
    step=0.1,
)

# Filter data based on selections
if selected_country == "All":
    filtered_data = zomato_data[zomato_data['Aggregate rating'] >= selected_rating]
else:
    filtered_data = zomato_data[ 
        (zomato_data['Country'] == selected_country) & 
        (zomato_data['Aggregate rating'] >= selected_rating)
    ]

# App title
st.title("Advanced Zomato Data Analysis")

# 1. Display Filtered Data
st.markdown("### Filtered Data")
st.dataframe(filtered_data)

# 2. Average Rating by Country (Static for All Countries)
st.markdown("### Average Rating by Country")
avg_rating_country = zomato_data.groupby('Country')['Aggregate rating'].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=avg_rating_country.index, y=avg_rating_country.values, palette='magma', ax=ax)
ax.set_title("Average Rating by Country (Static)")
ax.set_ylabel("Average Rating")
ax.set_xlabel("Country")
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)

# 3. Most Popular Cuisines
st.markdown("### Most Popular Cuisines")
top_cuisines = filtered_data['Cuisines'].str.split(', ').explode().value_counts().head(10)
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(y=top_cuisines.index, x=top_cuisines.values, palette='coolwarm', ax=ax)
ax.set_title("Top 10 Most Popular Cuisines")
ax.set_xlabel("Number of Restaurants")
ax.set_ylabel("Cuisine")
st.pyplot(fig)

# 4. Distribution of Price Ranges
st.markdown("### Distribution of Price Ranges Across Restaurants")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x='Price range', data=filtered_data, palette='Set2', ax=ax)
ax.set_title("Price Range Distribution")
ax.set_xlabel("Price Range")
ax.set_ylabel("Number of Restaurants")
st.pyplot(fig)

# 5. Average Rating by Price Range Across Cuisines (Using Heatmap)
st.markdown("### Average Rating by Price Range Across Cuisines")
# Limit cuisines to the top 10 based on frequency
top_cuisines_list = filtered_data['Cuisines'].value_counts().head(10).index
filtered_avg_rating_cuisines = filtered_data[filtered_data['Cuisines'].isin(top_cuisines_list)]

avg_rating_cuisines = (
    filtered_avg_rating_cuisines.groupby(['Cuisines', 'Price range'])['Aggregate rating']
    .mean()
    .reset_index()
    .sort_values(by='Aggregate rating', ascending=False)
)

# Pivot for heatmap
heatmap_data = avg_rating_cuisines.pivot_table(index='Cuisines', columns='Price range', values='Aggregate rating')

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt=".2f", ax=ax)
ax.set_title("Average Rating by Price Range and Cuisines (Top 10 Cuisines)", fontsize=16)
ax.set_xlabel("Price Range")
ax.set_ylabel("Cuisine")
st.pyplot(fig)

# 6. Cuisine Distribution for Highly Rated Restaurants by Price Range (Limited to Top 5 Cuisines)
st.markdown("### Cuisine Distribution for Highly Rated Restaurants by Price Range")
high_rated_data = filtered_data[filtered_data['Aggregate rating'] >= 4.5]

# Limit to fewer cuisines for simplicity (Top 5 most frequent cuisines in highly rated restaurants)
top_cuisines_high_rated = high_rated_data['Cuisines'].value_counts().head(5).index
high_rated_data_filtered = high_rated_data[high_rated_data['Cuisines'].isin(top_cuisines_high_rated)]

cuisine_price_dist = high_rated_data_filtered.groupby(['Price range', 'Cuisines']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(cuisine_price_dist, cmap="YlGnBu", ax=ax, cbar_kws={'label': 'Number of Restaurants'})
ax.set_title("Cuisine Distribution for Highly Rated Restaurants (Top 5 Cuisines)")
ax.set_xlabel("Cuisine")
ax.set_ylabel("Price Range")
st.pyplot(fig)

# 7. Correlation Between Cost for Two and Ratings Across Cuisines (Boxplot version)
st.markdown("### Correlation Between Cost for Two and Ratings Across Cuisines (Boxplot)")
# Limit cuisines to the top 10 most frequent
filtered_cost_data = filtered_data[filtered_data['Cuisines'].isin(top_cuisines_list)]

fig, ax = plt.subplots(figsize=(12, 8))
sns.boxplot(
    data=filtered_cost_data,
    x='Cuisines',
    y='Aggregate rating',
    hue='Price range',
    palette='Set2',
    ax=ax
)
ax.set_title("Cost vs. Ratings by Cuisine (Top 10 Cuisines) - Boxplot")
ax.set_xlabel("Cuisine")
ax.set_ylabel("Aggregate Rating")
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)

# 8. Conclusion Section
st.markdown("## Conclusions")
st.markdown("""
1. **Average Ratings by Country**: Certain countries, such as India and the UAE, have higher average ratings compared to others, reflecting better-rated dining experiences overall.
2. **Popular Cuisines**: Indian, Chinese, and Italian cuisines are among the most popular globally, as shown by the data.
3. **Price Range Distribution**: The majority of restaurants fall within moderate price ranges (Price Range 2 or 3), catering to mid-tier spending preferences.
4. **Highly Rated Restaurants and Cuisines**: Highly rated restaurants are distributed across cuisines, but premium price ranges often correlate with better ratings.
5. **Cost and Ratings**: There is a positive correlation between higher costs for two and aggregate ratings, particularly for fine dining cuisines like Italian and French.
""")
