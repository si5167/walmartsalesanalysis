# walmartsalesanalysis# Import 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings("ignore")

wm = pd.read_csv('C:/Users/siddhika/Downloads/WALMART_SALES_DATA.csv')
wm.head()

# Basic check up for missing value
wm.isna().sum()

#Task 1 - Which store has maximum sales
wm.groupby('Store').sum()['Weekly_Sales'].sort_values(ascending = False).head()

#Which store has maximum standard deviation i.e., the sales vary a lot. Also, find out the coefficient of mean to standard deviation
wm.groupby('Store').std()['Weekly_Sales'].sort_values(ascending = False).head()

# Calculating Coefficient of Variation (CV)

# Equation is CV = The Standard Deviation of dataset / The mean of dataset

cv = wm.groupby('Store').std()['Weekly_Sales'] / wm.groupby('Store').mean()['Weekly_Sales']
cv = cv.reset_index().rename(columns = {'Weekly_Sales': 'Coefficient of Variation'})

cv.head()

# Maximum CV
cv.sort_values(by='Coefficient of Variation', ascending = False).head()

#Task 3 - Which store/s has good quarterly growth rate in Q3’2012
#Convert Date column to datetime object
wm['Date'] = pd.to_datetime(wm['Date'], format="%d-%m-%Y")
wm.info()


# Extract the year and month
wm['Year'] = pd.DatetimeIndex(wm['Date']).year
wm['Month'] = pd.DatetimeIndex(wm['Date']).month
wm.head()

# Quarter Three is from month July (6) to September (9) and Year 2012

wm_q3_2012 = wm[(wm['Month'].isin([6,7,8,9])) & (wm['Year'] == 2012)] 
wm_q3_2012.head()

fig = px.bar(data_frame = wm_q3_2012.groupby('Store').sum().reset_index(),
             x = 'Store', y = 'Weekly_Sales', text = 'Weekly_Sales')


fig.update_layout(title = 'Total Weekly Sales of 45 Walmart stores during Q3 of 2012',
                  yaxis_title = 'Total Weekly Sales',
                  font = dict(family = "Courier New, monospace",
                              size = 14, color = 'black')
                  )

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(
    xaxis = dict(
        tickmode = 'array',
        tickvals = [n for n in range(1,46)],
    )
)



fig.show()

# Growth Rate by Store so first recorded date of quarter 3 to last date
wm_q3_2012['Date'].iloc[0] , wm_q3_2012['Date'].iloc[-1]

gr_wm = wm_q3_2012[(wm_q3_2012['Date'] == '2012-06-01') | (wm_q3_2012['Date'] == '2012-09-28')]
pct_wm = gr_wm.groupby('Store')['Weekly_Sales'].pct_change().dropna().reset_index().rename(columns={'index':'Store','Weekly_Sales':'%Change'})
pct_wm['Store'] = gr_wm['Store'].unique()
pct_wm.head()

# Top Performing WM Stores during Q3 2012
pct_wm.sort_values(by='%Change',ascending=False).head()

# Decrease in weekly sales a lot during Q3 2012
pct_wm.sort_values(by='%Change',ascending=False).tail()

#Some holidays have a negative impact on sales. Find out holidays which have higher sales than
#the mean sales in non-holiday season for all stores together

# Creating Holiday DataFrame
holiday = wm[wm['Holiday_Flag'] == 1]
holiday.tail()

# What are holiday dates present here?
holiday['Date'].value_counts()

# Super Bowl: 12-Feb-10, 11-Feb-11, 10-Feb-12, 8-Feb-13
# Labour Day: 10-Sep-10, 9-Sep-11, 7-Sep-12, 6-Sep-13
# Thanksgiving: 26-Nov-10, 25-Nov-11, 23-Nov-12, 29-Nov-13
# Christmas: 31-Dec-10, 30-Dec-11, 28-Dec-12, 27-Dec-13

from datetime import datetime

super_bowl = [datetime.strptime(date,"%d-%b-%y").date() for date in '12-Feb-10, 11-Feb-11, 10-Feb-12, 8-Feb-13'.split(", ")]
labour_day = [datetime.strptime(date,"%d-%b-%y").date() for date in '10-Sep-10, 9-Sep-11, 7-Sep-12, 6-Sep-13'.split(", ")]
thanksgiving = [datetime.strptime(date,"%d-%b-%y").date() for date in '26-Nov-10, 25-Nov-11, 23-Nov-12, 29-Nov-13'.split(", ")]
christmas = [datetime.strptime(date,"%d-%b-%y").date() for date in '31-Dec-10, 30-Dec-11, 28-Dec-12, 27-Dec-13'.split(", ")]

def assign_holiday(date):
    if date in super_bowl:
        return 'Super Bowl'
    elif date in labour_day:
        return 'Labor Day'
    elif date in thanksgiving:
        return 'Thanksgiving'
    elif date in christmas:
        return 'Christmas'
    else:
        return 'Not Holiday'
    
holiday['Occasion'] = holiday['Date'].apply(lambda date: assign_holiday(date))
holiday.head()

holiday_year = holiday.groupby(['Year','Occasion']).sum().reset_index()

fig = px.bar(data_frame = holiday_year, 
             x = 'Year', y = 'Weekly_Sales',
             color = 'Occasion', barmode = 'group',
             text = 'Weekly_Sales', height = 550,
             color_discrete_sequence = px.colors.qualitative.Safe)

fig.update_layout(title = 'Walmart Total Sales from 2010 to 2012 by Public Holiday',
                  yaxis_title = 'Total Sales',
                  legend_title = 'Holiday',
                  font = dict(family = "Courier New, monospace",
                              size = 14, color = 'black')
                  )

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.update_layout(
    xaxis = dict(
        tickmode = 'array',
        tickvals = [n for n in range(2010,2013)],
    )
)

fig.show()

# Mean sales in non-holiday season for all stores together
non_holi_mean_sales = wm[wm['Holiday_Flag'] == 0]['Weekly_Sales'].mean()
non_holi_mean_sales / 10**6

# Holiday Sales that is greater than mean 
holiday.groupby('Occasion')['Weekly_Sales'].mean() / 10**6 # Unit in Million (easier for comparison)

#Provide a monthly and semester view of sales in units and give insights
fig = px.line(data_frame = wm[wm['Store'].isin(wm.groupby('Store').sum().sort_values(by='Weekly_Sales',ascending = False).iloc[:3].index.to_list())],
              x = 'Date', y = 'Weekly_Sales',
              color = 'Store', color_discrete_sequence = px.colors.qualitative.Safe)

fig.update_layout(title = 'Top 3 Walmart Stores (by Total Sales) Weekly Sales',
                  yaxis_title = 'Weekly Sales',
                  font = dict(family = "Courier New, monospace",
                              size = 14, color = 'black')
                  )

fig.show()

# Monthly Sales
import calendar

fig = px.bar(data_frame = wm.groupby('Month').sum().reset_index(),
             x = 'Month', y = 'Weekly_Sales',
             text = 'Weekly_Sales', height = 550)

fig.update_layout(title = 'Walmart Overall Monthly Sales from 2011 to 2013',
                  yaxis_title = 'Total Sales',
                  font = dict(family = "Courier New, monospace",
                              size = 14, color = 'black')
                  )
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.update_yaxes(tickprefix="$")

fig.update_layout(
    xaxis = dict(
        ticktext = [calendar.month_name[n] for n in range(1,13)],
        tickvals = [n for n in range(1,13)]
    )
)

fig.show()

fig = px.bar(data_frame = wm.groupby(['Month','Year']).sum().reset_index(),
             x = 'Month', y = 'Weekly_Sales', color = 'Year',
             text = 'Weekly_Sales', height = 550)

fig.update_layout(title = 'Walmart Monthly Sales by Year',
                  yaxis_title = 'Total Sales',
                  font = dict(family = "Courier New, monospace",
                              size = 14, color = 'black')
                  )

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.update(layout_coloraxis_showscale=False)
fig.update_yaxes(tickprefix="$")

fig.update_layout(
    xaxis = dict(
        ticktext = [calendar.month_name[n] for n in range(1,13)],
        tickvals = [n for n in range(1,13)]
    )
)

fig.show()

# By Year Sales
plt.figure(dpi=120)
sns.barplot(data = wm.groupby('Year').sum().reset_index(),
            x = 'Year', y = 'Weekly_Sales', palette = 'Set2')
plt.title("Yearly Sales")
plt.ylabel("Sales (dollar)")
plt.show()

#statiscal model
wm.head()

# Adding More columns
wm['Day'] = pd.DatetimeIndex(wm['Date']).day
wm['Holiday'] = wm['Date'].apply(lambda date: assign_holiday(date))


wm.head()

# Checking for outlier and NaN value

features_list = 'Temperature, Fuel_Price, CPI, Unemployment, Year, Month, Day'.split(", ")

plt.figure(dpi=150)
count = 1
for feature in features_list:
    plt.subplot(4,2,count)
    sns.boxplot(wm[feature])
    count += 1
plt.tight_layout()
plt.show()

# Removing Outlier

def remove_out(feature):

    p25 = wm[feature].quantile(0.25)
    p75 = wm[feature].quantile(0.75)
    iqr = p75 - p25
    
    upper_limit = p75 + 1.5 * iqr 
    lower_limit = p25 - 1.5 * iqr
    
    new_df = wm[(wm[feature] > lower_limit) & (wm[feature] < upper_limit)]
    
    return new_df

for feature in features_list:
    wm = remove_out(feature)
wm.shape

from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
wm['Holiday'] = ordinal_encoder.fit_transform(wm[['Holiday']])

print(ordinal_encoder.categories_)

wm.head()

corr_matrix = wm.corr()
corr_matrix['Weekly_Sales'].sort_values(ascending = False)

from sklearn.model_selection import train_test_split

features = 'Temperature, Fuel_Price, CPI, Unemployment, Year, Month, Day, Holiday'.split(", ")
target = 'Weekly_Sales'

X = wm[features]
y = wm[target]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

prediction = lin_reg.predict(X_test)

from sklearn.metrics import mean_squared_error

lin_rmse = np.sqrt(mean_squared_error(y_test, prediction))
print("RSME:", lin_rmse)
print("Score:", lin_reg.score(X_train, y_train) * 100,"%")

sns.scatterplot(prediction, y_test)

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)

tree_prediction = tree_reg.predict(X_test)
tree_rmse = np.sqrt(mean_squared_error(y_test, tree_prediction))
print("RMSE:",tree_rmse)
print("Score:", tree_reg.score(X_train, y_train) * 100, "%")

sns.scatterplot(tree_prediction, y_test)

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(X_train, y_train)

forest_prediction = forest_reg.predict(X_test)
forest_rmse = np.sqrt(mean_squared_error(y_test, forest_prediction))
print("RMSE:",forest_rmse)
print("Score:", forest_reg.score(X_train, y_train) * 100, "%")

sns.scatterplot(forest_prediction, y_test)

