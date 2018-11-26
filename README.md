
## 1. Introduction

My [last project](https://app.powerbi.com/view?r=eyJrIjoiNzMxZTliYmUtYzI3My00NjZkLTlkM2MtOGEyYWYzMDRmMzI5IiwidCI6Ijg2MGEzN2E4LTJjODEtNDk1Mi04ZWQwLTk5MmFjNTFlMmJkZiIsImMiOjN9) provided a broad overview of the resale real estate market in the GTA and helped to visualize what was going on in terms of pricing and supply. This project pivots to a more fine-grained level and looks at individual condominium apartment listings, with the goal of building a price predictor using machine learning models.


### i. Background and Methodology

Condo listings data was scraped using Scrapy from a third-party listings website in October 2018. It's important to note that it's common practice for boutique or luxury condo buildings to be exclusively listed by certain brokerages. These higher-end listings may not be posted publicly, and thus are not captured in the data.

As the real estate dashboard from my previous project showed, sale price growth accelerated between 2012 and 2017, so prices before 2017 would not necessarily aid our model. Also, to maintain data purity, we don't necessarily want listings with a 'terminated' or 'expired' status - just listings that have sold as our target variable is sold price. As such, this project wittles down our sample size to almost 44,000 sold listings, transacted between January 2017 and October 2018.

Listings data provides crucial information such as size range and number of bedrooms. However, I was curious about pricing's relationship with other data points, such as building age and size. I also wanted to see the influence of proximity to higher-order transit. For the purposes of this project, I define this as subway stations, streetcar stops, GO train stations, and Bus Rapid Transit (BRT)\* stops. Using Open Data portals (and General Transit Feed Specification data to fill in gaps for tricky streetcar stops data) and with a bit of wrangling in QGIS and database joining, I included variables for four features: distance to nearest BRT stop, nearest Subway or Streetcar stop, nearest GO train, and overall nearest higher-order transit stop (i.e. the closest of the three).

\* _includes three routes with dedicated right-of-ways: York Region VIVA BRT on Davis Drive, York Region Highway 7 East BRT, and Mississauga MiWay (dedicated ROW route only)._

Other variables I've included are the presence of certain amenities (e.g. pool, gym, party room), total amenities, and municipality. A full list will follow shortly.


### ii. Before we dive in...

I should note that the bulk of this post, particularly at the beginning, will be on the data cleaning/preprocessing and encoding. Real estate data (at least in the Greater Toronto Area) is more often than not incomplete and "dirty", sometimes unusable. However, with certain data parsing methods, we can fill in these gaps.

An example of this would be listings with missing size range information. However, instead of throwing these datapoints in the recycling bin, using regular expressions, size data can be extracted from listings descriptions, wherein realtors would describe the unit and sometimes mention its square footage. Same goes for number of bedrooms. So from these descriptions, I have parsed out size and number of bedrooms into **Size_Description** and ** Beds_Description** fields, respectively.

Other things worth noting:
- Certain columns like "Amenities" are binary-encoded (i.e. 1 = True, 0 = False)
- In our dataframes, "nan" is Pythonic for "not a number" and indicates missing data


```python
# Libraries to load
import time
import json
import random
import os
import csv
import re

import pandas as pd
import numpy as np

from collections import Counter

# For graphing
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# Display options
pd.set_option('display.float_format', lambda x: '%.4f' % x)
pd.set_option("display.max_columns", 50)
```


```python
# Load and view a sample of the data
with open("output_resale_2018-10-31_clean.csv", 'r') as f:
    resales_df = pd.read_csv(f, index_col = False)
resales_df.sample(20)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Amenities__Gym / Exercise Room</th>
      <th>Amenities__Party Room</th>
      <th>Amenities__Pool</th>
      <th>Balcony</th>
      <th>Bath</th>
      <th>Beds</th>
      <th>Beds_Description</th>
      <th>Building_Reg_Date</th>
      <th>Building_Completion</th>
      <th>Building_Storeys</th>
      <th>Building_Units</th>
      <th>Days on Market</th>
      <th>Den_Description</th>
      <th>Locker</th>
      <th>Muni</th>
      <th>Nearest_BRT</th>
      <th>Nearest_GO</th>
      <th>Nearest_HOT</th>
      <th>Nearest_SS</th>
      <th>Parking</th>
      <th>Size_Description</th>
      <th>Size_Range</th>
      <th>Sold Price</th>
      <th>Unit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33132</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Open</td>
      <td>2.0000</td>
      <td>2</td>
      <td>2.0000</td>
      <td>2016-10-28</td>
      <td>nan</td>
      <td>11</td>
      <td>367</td>
      <td>6.0000</td>
      <td>0</td>
      <td>Owned</td>
      <td>toronto</td>
      <td>21243.2845</td>
      <td>2120.2162</td>
      <td>118.1565</td>
      <td>118.1565</td>
      <td>Yes</td>
      <td>0</td>
      <td>700-799</td>
      <td>665000.0000</td>
      <td>N1006</td>
    </tr>
    <tr>
      <th>15484</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Open</td>
      <td>2.0000</td>
      <td>2+1</td>
      <td>nan</td>
      <td>2017-01-13</td>
      <td>nan</td>
      <td>23</td>
      <td>399</td>
      <td>0.0000</td>
      <td>1</td>
      <td>Owned</td>
      <td>toronto</td>
      <td>17052.9692</td>
      <td>4936.1694</td>
      <td>80.3128</td>
      <td>80.3128</td>
      <td>Yes</td>
      <td>NaN</td>
      <td>1200-1399</td>
      <td>1315000.0000</td>
      <td>1417</td>
    </tr>
    <tr>
      <th>30261</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Open</td>
      <td>1.0000</td>
      <td>1+1</td>
      <td>nan</td>
      <td>2011-04-29</td>
      <td>nan</td>
      <td>41</td>
      <td>435</td>
      <td>7.0000</td>
      <td>0</td>
      <td>No</td>
      <td>toronto</td>
      <td>8630.8855</td>
      <td>731.5015</td>
      <td>731.5015</td>
      <td>1333.2279</td>
      <td>Yes</td>
      <td>0</td>
      <td>600-699</td>
      <td>356000.0000</td>
      <td>2607</td>
    </tr>
    <tr>
      <th>11908</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>None</td>
      <td>2.0000</td>
      <td>2</td>
      <td>2.0000</td>
      <td>2002-01-22</td>
      <td>nan</td>
      <td>8</td>
      <td>64</td>
      <td>36.0000</td>
      <td>0</td>
      <td>Owned</td>
      <td>mississauga</td>
      <td>886.2491</td>
      <td>3057.7813</td>
      <td>886.2491</td>
      <td>14714.0617</td>
      <td>Yes</td>
      <td>0</td>
      <td>800-899</td>
      <td>290000.0000</td>
      <td>104</td>
    </tr>
    <tr>
      <th>9558</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Open</td>
      <td>1.0000</td>
      <td>1+1</td>
      <td>1.0000</td>
      <td>2012-04-18</td>
      <td>nan</td>
      <td>28</td>
      <td>1131</td>
      <td>60.0000</td>
      <td>1</td>
      <td>Owned</td>
      <td>toronto</td>
      <td>8298.3440</td>
      <td>858.1979</td>
      <td>157.1629</td>
      <td>157.1629</td>
      <td>Yes</td>
      <td>739</td>
      <td>700-799</td>
      <td>460000.0000</td>
      <td>1206</td>
    </tr>
    <tr>
      <th>5808</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>None</td>
      <td>2.0000</td>
      <td>3</td>
      <td>3.0000</td>
      <td>1990-11-12</td>
      <td>nan</td>
      <td>24</td>
      <td>235</td>
      <td>5.0000</td>
      <td>0</td>
      <td>Owned</td>
      <td>brampton</td>
      <td>12494.3652</td>
      <td>3358.3294</td>
      <td>3358.3294</td>
      <td>17843.2611</td>
      <td>Yes</td>
      <td>0</td>
      <td>1400-1599</td>
      <td>430000.0000</td>
      <td>210</td>
    </tr>
    <tr>
      <th>417</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>Open</td>
      <td>1.0000</td>
      <td>2</td>
      <td>2.0000</td>
      <td>2015-12-02</td>
      <td>nan</td>
      <td>11</td>
      <td>96</td>
      <td>44.0000</td>
      <td>0</td>
      <td>Owned</td>
      <td>toronto</td>
      <td>21135.7993</td>
      <td>1159.5990</td>
      <td>94.7533</td>
      <td>94.7533</td>
      <td>Yes</td>
      <td>NaN</td>
      <td>800-899</td>
      <td>685000.0000</td>
      <td>403</td>
    </tr>
    <tr>
      <th>43739</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>None</td>
      <td>2.0000</td>
      <td>2+1</td>
      <td>2.0000</td>
      <td>1981-03-04</td>
      <td>nan</td>
      <td>24</td>
      <td>230</td>
      <td>20.0000</td>
      <td>0</td>
      <td>Ensuite</td>
      <td>toronto</td>
      <td>6036.0944</td>
      <td>1616.6446</td>
      <td>1616.6446</td>
      <td>2195.5561</td>
      <td>Yes</td>
      <td>1,711</td>
      <td>1600-1799</td>
      <td>675000.0000</td>
      <td>307</td>
    </tr>
    <tr>
      <th>5601</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Open</td>
      <td>2.0000</td>
      <td>NaN</td>
      <td>2.0000</td>
      <td>2014-07-21</td>
      <td>nan</td>
      <td>4</td>
      <td>125</td>
      <td>16.0000</td>
      <td>0</td>
      <td>No</td>
      <td>brampton</td>
      <td>17509.3515</td>
      <td>241.7032</td>
      <td>241.7032</td>
      <td>23755.3748</td>
      <td>Yes</td>
      <td>0</td>
      <td>unknown</td>
      <td>415000.0000</td>
      <td>301</td>
    </tr>
    <tr>
      <th>16240</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Open</td>
      <td>1.0000</td>
      <td>2</td>
      <td>nan</td>
      <td>2005-10-11</td>
      <td>nan</td>
      <td>4</td>
      <td>48</td>
      <td>0.0000</td>
      <td>0</td>
      <td>Exclusive</td>
      <td>burlington</td>
      <td>19593.0060</td>
      <td>3142.5288</td>
      <td>3142.5288</td>
      <td>31575.4287</td>
      <td>Yes</td>
      <td>0</td>
      <td>800-899</td>
      <td>375000.0000</td>
      <td>212</td>
    </tr>
    <tr>
      <th>28140</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Open</td>
      <td>1.0000</td>
      <td>1</td>
      <td>nan</td>
      <td>2014-05-05</td>
      <td>nan</td>
      <td>42</td>
      <td>318</td>
      <td>6.0000</td>
      <td>0</td>
      <td>Owned</td>
      <td>toronto</td>
      <td>20734.3602</td>
      <td>1256.2569</td>
      <td>96.2079</td>
      <td>96.2079</td>
      <td>No</td>
      <td>NaN</td>
      <td>0-499</td>
      <td>508000.0000</td>
      <td>3006</td>
    </tr>
    <tr>
      <th>22401</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Open</td>
      <td>1.0000</td>
      <td>1+1</td>
      <td>1.0000</td>
      <td>2013-07-02</td>
      <td>nan</td>
      <td>34</td>
      <td>525</td>
      <td>3.0000</td>
      <td>1</td>
      <td>No</td>
      <td>toronto</td>
      <td>9972.3615</td>
      <td>2875.0427</td>
      <td>23.1734</td>
      <td>23.1734</td>
      <td>Yes</td>
      <td>0</td>
      <td>700-799</td>
      <td>425000.0000</td>
      <td>1714</td>
    </tr>
    <tr>
      <th>16360</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Terrace</td>
      <td>2.0000</td>
      <td>2</td>
      <td>nan</td>
      <td>2007-05-09</td>
      <td>nan</td>
      <td>8</td>
      <td>199</td>
      <td>3.0000</td>
      <td>0</td>
      <td>Owned</td>
      <td>toronto</td>
      <td>13826.7975</td>
      <td>5119.6335</td>
      <td>2300.5660</td>
      <td>2300.5660</td>
      <td>Yes</td>
      <td>1522</td>
      <td>1400-1599</td>
      <td>1240000.0000</td>
      <td>209</td>
    </tr>
    <tr>
      <th>9656</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Open</td>
      <td>2.0000</td>
      <td>1+1</td>
      <td>1.0000</td>
      <td>2012-04-18</td>
      <td>nan</td>
      <td>28</td>
      <td>1131</td>
      <td>25.0000</td>
      <td>1</td>
      <td>Owned</td>
      <td>toronto</td>
      <td>8298.3440</td>
      <td>858.1979</td>
      <td>157.1629</td>
      <td>157.1629</td>
      <td>Yes</td>
      <td>720</td>
      <td>700-799</td>
      <td>458000.0000</td>
      <td>815</td>
    </tr>
    <tr>
      <th>26069</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>Open</td>
      <td>1.0000</td>
      <td>1+1</td>
      <td>nan</td>
      <td>2016-10-28</td>
      <td>nan</td>
      <td>39</td>
      <td>405</td>
      <td>21.0000</td>
      <td>1</td>
      <td>No</td>
      <td>toronto</td>
      <td>21272.6972</td>
      <td>1086.9472</td>
      <td>112.9659</td>
      <td>112.9659</td>
      <td>No</td>
      <td>652</td>
      <td>600-699</td>
      <td>544000.0000</td>
      <td>1103</td>
    </tr>
    <tr>
      <th>33557</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Open</td>
      <td>1.0000</td>
      <td>1+1</td>
      <td>nan</td>
      <td>2016-07-21</td>
      <td>nan</td>
      <td>22</td>
      <td>434</td>
      <td>1.0000</td>
      <td>0</td>
      <td>Owned</td>
      <td>toronto</td>
      <td>12955.4575</td>
      <td>1347.6351</td>
      <td>120.1811</td>
      <td>120.1811</td>
      <td>Yes</td>
      <td>0</td>
      <td>500-599</td>
      <td>530109.0000</td>
      <td>2104</td>
    </tr>
    <tr>
      <th>36134</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Open</td>
      <td>2.0000</td>
      <td>3</td>
      <td>0.0000</td>
      <td>1979-03-05</td>
      <td>nan</td>
      <td>15</td>
      <td>177</td>
      <td>7.0000</td>
      <td>0</td>
      <td>No</td>
      <td>brampton</td>
      <td>11651.8852</td>
      <td>2701.4469</td>
      <td>2701.4469</td>
      <td>17173.3701</td>
      <td>Yes</td>
      <td>0</td>
      <td>1200-1399</td>
      <td>300000.0000</td>
      <td>809</td>
    </tr>
    <tr>
      <th>32710</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Open</td>
      <td>2.0000</td>
      <td>2</td>
      <td>nan</td>
      <td>2017-04-07</td>
      <td>nan</td>
      <td>42</td>
      <td>393</td>
      <td>1.0000</td>
      <td>0</td>
      <td>Owned</td>
      <td>toronto</td>
      <td>21389.1675</td>
      <td>944.9315</td>
      <td>154.9281</td>
      <td>154.9281</td>
      <td>Yes</td>
      <td>0</td>
      <td>800-899</td>
      <td>993800.0000</td>
      <td>2905</td>
    </tr>
    <tr>
      <th>9653</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Open</td>
      <td>1.0000</td>
      <td>1+1</td>
      <td>1.0000</td>
      <td>2012-04-18</td>
      <td>nan</td>
      <td>28</td>
      <td>1131</td>
      <td>8.0000</td>
      <td>1</td>
      <td>Owned</td>
      <td>toronto</td>
      <td>8298.3440</td>
      <td>858.1979</td>
      <td>157.1629</td>
      <td>157.1629</td>
      <td>Yes</td>
      <td>NaN</td>
      <td>600-699</td>
      <td>419000.0000</td>
      <td>1003</td>
    </tr>
    <tr>
      <th>27831</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Open</td>
      <td>2.0000</td>
      <td>1</td>
      <td>nan</td>
      <td>2000-01-06</td>
      <td>nan</td>
      <td>13</td>
      <td>207</td>
      <td>2.0000</td>
      <td>0</td>
      <td>Owned</td>
      <td>toronto</td>
      <td>21174.3089</td>
      <td>838.2169</td>
      <td>83.6455</td>
      <td>83.6455</td>
      <td>Yes</td>
      <td>0</td>
      <td>900-999</td>
      <td>655000.0000</td>
      <td>PH1</td>
    </tr>
  </tbody>
</table>
</div>



### Remove Parking, Locker, and Wine Cellar Listings
As mentioned, real estate listings are highly nuanced and require a lot of cleaning. In our scraped data, there are listings for parking spots, lockers, and even wine cellars! These are less than $10,000 - a price threshold which no condo listing falls under (through a separate, manual exploration). So let's remove them from our data.


```python
resales_df = resales_df[resales_df['Sold Price'] > 10000]
resales_df.reset_index(drop=True, inplace=True)
```

### How much data is useful / useless?


```python
resales_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 43941 entries, 0 to 43940
    Data columns (total 24 columns):
    Amenities__Gym / Exercise Room    43941 non-null int64
    Amenities__Party Room             43941 non-null int64
    Amenities__Pool                   43941 non-null int64
    Balcony                           43941 non-null object
    Bath                              43917 non-null float64
    Beds                              42482 non-null object
    Beds_Description                  27084 non-null float64
    Building_Reg_Date                 43753 non-null object
    Building_Completion               61 non-null float64
    Building_Storeys                  43941 non-null int64
    Building_Units                    43941 non-null int64
    Days on Market                    43931 non-null float64
    Den_Description                   43941 non-null int64
    Locker                            43941 non-null object
    Muni                              43941 non-null object
    Nearest_BRT                       43941 non-null float64
    Nearest_GO                        43941 non-null float64
    Nearest_HOT                       43941 non-null float64
    Nearest_SS                        43941 non-null float64
    Parking                           43941 non-null object
    Size_Description                  40541 non-null object
    Size_Range                        43941 non-null object
    Sold Price                        43941 non-null float64
    Unit                              43039 non-null object
    dtypes: float64(9), int64(6), object(9)
    memory usage: 8.0+ MB
    

As you can quickly see above, of the 43,941 listings, almost all fields have values. 

It's worth pointing on that the most incomplete field is **Building_Completion**, for which only 61 listings have values. However, I have deliberately included this as we understand that sometimes there are listings for units in buildings that may have not yet registered (i.e. under construction) but will register in the near future. I'll confess this was included only after digging into the data and realizing that some listings with no **Building_Reg_Date** data were not necessarily missing this data, but in fact were not yet completed. 

There was a lot of this kind of unseen back and forth in this project. :(

From the sample we can also see there are a handful of 'unknown' string values, which are essentially nulls. We will need to replace them with an actual null value understood by our model: _NaN_, which stands for Not a Number. After some cleaning, we can visualize how much data we are actually missing.


```python
# Replace null or unknown values with np.nan
resales_df['Balcony'].replace("Unknown", np.nan, inplace=True)
resales_df['Size_Range'].replace("unknown", np.nan, inplace=True)
resales_df['Size_Description'] = resales_df['Size_Description'].str.replace(",", "").replace("0",np.nan).replace("",np.nan).astype(float)

# Visualize missing data:
variables = resales_df.columns
count = []
for var in variables:
    notnull = resales_df[var].count()
    count.append(notnull)
count_pct = np.round(pd.Series(count) * 100 / len(resales_df), 2)

# Plot
plt.figure(figsize=(20,12))
plt.barh(width = count_pct, y = range(len(variables)))
plt.yticks(range(len(variables)), variables, fontsize = 18)
plt.title('% of Available (non-null) Data', fontsize=25)
plt.show()
```


![png](output_8_0.png)


Although there is next to no information for **Building_Completion**, we will retain this as it will help us fill in gaps in **Building_Reg_Date** later.

Exploring our data a bit further, if we were to use only rows without nulls, how many rows would we have? Let's omit **Size Description, Beds Description, Building_Completion** as they have the most nulls.


```python
no_nulls = resales_df.drop(['Size_Description', 'Beds_Description','Building_Completion'], axis=1).dropna()
print("From an original {:,} listings, there are {:,} listings (or {:.2f}%) that have no nulls at all.".format(\
    len(resales_df), len(no_nulls), len(no_nulls)/len(resales_df)*100))
```

    From an original 43,941 listings, there are 42,028 listings (or 95.65%) that have no nulls at all.
    

For the remaining percentage of listings that have null data, we can either drop them or impute data to fill them in. We'll do this in the following section where we'll simultaneously encode our data. 

## 2. Encoding Fields:
##### 1. Size_Range / Size_Description
##### 2. Sold_Price 
##### 3. Balcony
##### 4. Municipality
##### 5. Building_Storeys
##### 6. Building_Units
##### 7. Building_Reg_Date
##### 8. Beds_Description vs. Beds
##### 9. Baths
##### 10. Days on Market
##### 11. Nearest Higher-Order Transit Stop
##### 12. Parking
##### 13. Lockers
##### 14. Unit (Construct two new binary features: Is_TH, Is_PH)

---


### 2.1 Size_Range / Size_Description

Assuming that size is a very important predictor of price, we would like to retain as much **Size_Range** info as possible. Are there any rows that have **Size_Description** info extracted but no **Size_Range** info which we could fill in?


```python
sizes = resales_df[resales_df['Size_Range'].isnull()]['Size_Description'].notnull()

print("There are {} listings that do not have Size_Range but have sizes from Size Description.".format(sum(sizes)))
```

    There are 286 listings that do not have Size_Range but have sizes from Size Description.
    

Our machines won't understand that "500-599" is a range of sizes smaller than "1800-1999". To assign meaning and hierarchy to this, we will map size ranges into easily understandable levels from 1 (0-499 sq. ft.) ... to 24 (5,000 sq ft. +).


```python
# First we build a function that will help us encode unit sizes into size range level.
def codify_sizes(input_size):
    # Note this is for individual sizes, not ranges.
    input_size = float(input_size)
### Conversion 
    if input_size < 250:
        size_range = 0
        # probably parking or locker...
    elif input_size <= 499:
        size_range = 1
    elif input_size <= 599:
        size_range = 2
    elif input_size <= 699:
        size_range = 3
    elif input_size <= 799:
        size_range = 4
    elif input_size <= 899:
        size_range = 5
    elif input_size <= 999:
        size_range = 6
    elif input_size <= 1199:
        size_range = 7
    elif input_size <= 1399:
        size_range = 8
    elif input_size <= 1599:
        size_range = 9
    elif input_size <= 1799:
        size_range = 10
    elif input_size <= 1999:
        size_range = 11
    elif input_size <= 2249:
        size_range = 12
    elif input_size <= 2499:
        size_range = 13
    elif input_size <= 2749:
        size_range = 14
    elif input_size <= 2999:
        size_range = 15
    elif input_size <= 3249:
        size_range = 16
    elif input_size <= 3499:
        size_range = 17
    elif input_size <= 3749:
        size_range = 18
    elif input_size <= 3999:
        size_range = 19
    elif input_size <= 4249:
        size_range = 20
    elif input_size <= 4499:
        size_range = 21
    elif input_size <= 4749:
        size_range = 22
    elif input_size <= 4999:
        size_range = 23
    elif input_size >= 5000:
        size_range = 24
    else:
        print("input size is not a number, please check.")
        
    return size_range

    

# Now we build a function that encodes size ranges into size range level
def codify_size_range(input_range):
    input_range = str(input_range)
    range_dict = {'0-499': 1, '500-599': 2, '600-699': 3,'700-799': 4,'800-899': 5,
                  '900-999': 6,'1000-1199': 7,'1200-1399': 8,'1400-1599': 9,
                  '1600-1799': 10,'1800-1999': 11,'2000-2249': 12,'2250-2499': 13,
                  '2500-2749': 14,'2750-2999': 15,'3000-3249': 16,'3250-3499': 17,
                  '3500-3749': 18,'3750-3999': 19,'4000-4249': 20,'4250-449': 21,
                  '4500-4749': 22,'4750-4999': 23, '5000 +': 24, 'nan': np.nan}
    if input_range in range_dict:
        x = range_dict[input_range]
    else:
        start_range = re.findall('^[0-9]*',input_range)[0]
        end_range = re.findall('[0-9]*$',input_range)[0]
        med = (int(start_range) + int(end_range)) / 2
        x = codify_sizes(med)
    return x
```

Now we run the function on each listing.


```python
# Refactor Size_Range so that they fall into their factor levels.
resales_df['Size_Range'] = [codify_size_range(i) for i in resales_df['Size_Range']]

# Fill null Size_Range with size descriptions
replace_rows = resales_df['Size_Range'].isnull() & resales_df['Size_Description'].notnull()
resales_df.loc[replace_rows,'Size_Range'] = [codify_sizes(i) for i in resales_df.loc[replace_rows, 'Size_Description']]

resales_df.drop('Size_Description', axis=1, inplace=True)

# Drop Where Size Range is null.
len_before = sum(resales_df['Size_Range'].isnull())
resales_df.dropna(subset = ['Size_Range'], axis = 0, inplace = True)
print("{:,} listings were dropped for not having size_range data. New number of listings is {:,}.".format(len_before, len(resales_df)))

# Reset Index
resales_df.reset_index(drop = True, inplace = True)
```

    1,174 listings were dropped for not having size_range data. New number of listings is 42,767.
    

### 2.2 Sold Price
This is our target variable! Let's explore its distribution in our data.


```python
resales_df['Sold Price'].describe()
```




    count      42767.0000
    mean      538109.4844
    std       300302.1524
    min        89500.0000
    25%       390000.0000
    50%       470000.0000
    75%       600000.0000
    max     11500000.0000
    Name: Sold Price, dtype: float64




```python
fig = plt.figure(figsize=(12,6))
plt.title("Distribution of Listings by Sold Price", fontsize = 16)
plt.ylabel('Frequency')
ax = sns.distplot(resales_df['Sold Price'])
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('${x:,.0f}'))

plt.show()
```

    C:\Users\Fabienne\Anaconda3\lib\site-packages\matplotlib\axes\_axes.py:6521: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      alternative="'density'", removal="3.1")
    


![png](output_20_1.png)


As we can see from the distribution, luxury units (up to \$12 million! ) are skewing our distribution to the right. Depending on what model we use, this could be a problem. For visualization purposes, let's temporarily omit listings over $2 million as these are less common.


```python
fig = plt.figure(figsize=(12,8))
plt.title("Distribution of Listings with Sold Price under $2 Million", fontsize = 16)
plt.ylabel('Frequency')
ax = sns.distplot(resales_df['Sold Price'], bins = 400)
plt.xlim(0,2000000)
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('${x:,.0f}'))
plt.show()
```

    C:\Users\Fabienne\Anaconda3\lib\site-packages\matplotlib\axes\_axes.py:6521: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      alternative="'density'", removal="3.1")
    


![png](output_22_1.png)


This looks about right - listings typically hover between \$300,000 to \$600,000.

### 2.3 Balcony
We're going to One-Hot encode our Balcony data. We can't factor into levels like we did with size ranges because the data isn't orginal. If 'Open Balcony' was 1 and 'Juliette Balcony' was 2, it doesn't mean Juliettes are "bigger" or "better" than Open balconies. Thus we'll use One-Hot Encoding to create fields for each balcony type, with 0 indicating if it isn't that type, and 1 if it is.


```python
Counter(resales_df['Balcony'])
```




    Counter({'Open': 29602,
             'None': 6897,
             'Terrace': 4009,
             'Juliette': 976,
             'Enclosed': 1280,
             nan: 3})



Given that there are just 3 missing 'Balcony' values, we can avoid removing the entire observation and instead replace it with the most common observation, 'Open'.


```python
resales_df.loc[resales_df['Balcony'].isnull(), 'Balcony'] = 'Open'
```


```python
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Data Scaler
from sklearn.preprocessing import StandardScaler

# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(resales_df['Balcony'])

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

one_hot_labels = ['Balcony_'+i for i in label_encoder.classes_]
one_hot_balcony = pd.DataFrame(onehot_encoded, columns = one_hot_labels)


# Delete the old columns
resales_df = resales_df.drop('Balcony', axis = 1)

# Add the new One-Hot encoded columns
resales_df = pd.concat([resales_df, one_hot_balcony], axis = 1)

resales_df.sample(6)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Amenities__Gym / Exercise Room</th>
      <th>Amenities__Party Room</th>
      <th>Amenities__Pool</th>
      <th>Bath</th>
      <th>Beds</th>
      <th>Beds_Description</th>
      <th>Building_Reg_Date</th>
      <th>Building_Completion</th>
      <th>Building_Storeys</th>
      <th>Building_Units</th>
      <th>Days on Market</th>
      <th>Den_Description</th>
      <th>Locker</th>
      <th>Muni</th>
      <th>Nearest_BRT</th>
      <th>Nearest_GO</th>
      <th>Nearest_HOT</th>
      <th>Nearest_SS</th>
      <th>Parking</th>
      <th>Size_Range</th>
      <th>Sold Price</th>
      <th>Unit</th>
      <th>Balcony_Enclosed</th>
      <th>Balcony_Juliette</th>
      <th>Balcony_None</th>
      <th>Balcony_Open</th>
      <th>Balcony_Terrace</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>36529</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1.0000</td>
      <td>1</td>
      <td>1.0000</td>
      <td>2001-06-27</td>
      <td>nan</td>
      <td>18</td>
      <td>91</td>
      <td>0.0000</td>
      <td>0</td>
      <td>Exclusive</td>
      <td>toronto</td>
      <td>18770.7014</td>
      <td>3104.1265</td>
      <td>112.9149</td>
      <td>112.9149</td>
      <td>Yes</td>
      <td>3.0000</td>
      <td>480000.0000</td>
      <td>1507</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>2154</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1.0000</td>
      <td>2+1</td>
      <td>nan</td>
      <td>1971-06-12</td>
      <td>nan</td>
      <td>26</td>
      <td>897</td>
      <td>10.0000</td>
      <td>1</td>
      <td>Exclusive</td>
      <td>toronto</td>
      <td>4086.6102</td>
      <td>1280.5084</td>
      <td>1280.5084</td>
      <td>6110.9569</td>
      <td>Yes</td>
      <td>7.0000</td>
      <td>260000.0000</td>
      <td>2101</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>1006</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1.0000</td>
      <td>1+1</td>
      <td>1.0000</td>
      <td>1980-09-11</td>
      <td>nan</td>
      <td>5</td>
      <td>41</td>
      <td>2.0000</td>
      <td>1</td>
      <td>Ensuite</td>
      <td>toronto</td>
      <td>18782.0425</td>
      <td>3413.9517</td>
      <td>319.8708</td>
      <td>319.8708</td>
      <td>Yes</td>
      <td>5.0000</td>
      <td>480000.0000</td>
      <td>505</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>26239</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1.0000</td>
      <td>1+1</td>
      <td>nan</td>
      <td>2009-04-28</td>
      <td>nan</td>
      <td>24</td>
      <td>207</td>
      <td>12.0000</td>
      <td>0</td>
      <td>Owned</td>
      <td>toronto</td>
      <td>5347.7286</td>
      <td>1185.6612</td>
      <td>367.8915</td>
      <td>367.8915</td>
      <td>Yes</td>
      <td>3.0000</td>
      <td>375000.0000</td>
      <td>1204</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>13322</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2.0000</td>
      <td>2</td>
      <td>nan</td>
      <td>2005-05-12</td>
      <td>nan</td>
      <td>31</td>
      <td>398</td>
      <td>17.0000</td>
      <td>0</td>
      <td>Common</td>
      <td>toronto</td>
      <td>7150.1683</td>
      <td>3762.7534</td>
      <td>566.6894</td>
      <td>566.6894</td>
      <td>Yes</td>
      <td>6.0000</td>
      <td>639000.0000</td>
      <td>1411</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>30794</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1.0000</td>
      <td>1+1</td>
      <td>1.0000</td>
      <td>2015-09-28</td>
      <td>nan</td>
      <td>20</td>
      <td>257</td>
      <td>0.0000</td>
      <td>1</td>
      <td>No</td>
      <td>toronto</td>
      <td>8427.3220</td>
      <td>747.9461</td>
      <td>264.3569</td>
      <td>264.3569</td>
      <td>Yes</td>
      <td>3.0000</td>
      <td>435000.0000</td>
      <td>716</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>



### 2.4 Municipality (One-Hot Encoding)
Repeat the One-Hot encoding process for "Municipality" field.


```python
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(resales_df['Muni'])

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

one_hot_muni = pd.DataFrame(onehot_encoded, columns = label_encoder.classes_)

# Delete the old columns
resales_df = resales_df.drop('Muni', axis = 1)

# Add the new One-Hot encoded columns
resales_df = pd.concat([resales_df, one_hot_muni], axis = 1)

resales_df.sample(6)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Amenities__Gym / Exercise Room</th>
      <th>Amenities__Party Room</th>
      <th>Amenities__Pool</th>
      <th>Bath</th>
      <th>Beds</th>
      <th>Beds_Description</th>
      <th>Building_Reg_Date</th>
      <th>Building_Completion</th>
      <th>Building_Storeys</th>
      <th>Building_Units</th>
      <th>Days on Market</th>
      <th>Den_Description</th>
      <th>Locker</th>
      <th>Nearest_BRT</th>
      <th>Nearest_GO</th>
      <th>Nearest_HOT</th>
      <th>Nearest_SS</th>
      <th>Parking</th>
      <th>Size_Range</th>
      <th>Sold Price</th>
      <th>Unit</th>
      <th>Balcony_Enclosed</th>
      <th>Balcony_Juliette</th>
      <th>Balcony_None</th>
      <th>Balcony_Open</th>
      <th>Balcony_Terrace</th>
      <th>ajax</th>
      <th>aurora</th>
      <th>brampton</th>
      <th>burlington</th>
      <th>markham</th>
      <th>milton</th>
      <th>mississauga</th>
      <th>newmarket</th>
      <th>oakville</th>
      <th>oshawa</th>
      <th>pickering</th>
      <th>richmondhill</th>
      <th>toronto</th>
      <th>vaughan</th>
      <th>whitby</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9305</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1.0000</td>
      <td>1</td>
      <td>1.0000</td>
      <td>2002-07-19</td>
      <td>nan</td>
      <td>9</td>
      <td>183</td>
      <td>7.0000</td>
      <td>0</td>
      <td>Owned</td>
      <td>13700.6722</td>
      <td>6437.4584</td>
      <td>6437.4584</td>
      <td>13595.3183</td>
      <td>Yes</td>
      <td>1.0000</td>
      <td>228800.0000</td>
      <td>405</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>20251</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2.0000</td>
      <td>2</td>
      <td>2.0000</td>
      <td>2013-01-10</td>
      <td>nan</td>
      <td>18</td>
      <td>196</td>
      <td>19.0000</td>
      <td>0</td>
      <td>Owned</td>
      <td>1172.9507</td>
      <td>2638.1298</td>
      <td>1172.9507</td>
      <td>14261.3812</td>
      <td>Yes</td>
      <td>5.0000</td>
      <td>490000.0000</td>
      <td>PH3</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>13757</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2.0000</td>
      <td>2+1</td>
      <td>nan</td>
      <td>2006-05-01</td>
      <td>nan</td>
      <td>37</td>
      <td>597</td>
      <td>0.0000</td>
      <td>1</td>
      <td>Owned</td>
      <td>16148.7624</td>
      <td>977.7859</td>
      <td>195.0677</td>
      <td>195.0677</td>
      <td>Yes</td>
      <td>6.0000</td>
      <td>883000.0000</td>
      <td>3001</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>162</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2.0000</td>
      <td>2</td>
      <td>nan</td>
      <td>1987-01-29</td>
      <td>nan</td>
      <td>19</td>
      <td>300</td>
      <td>7.0000</td>
      <td>0</td>
      <td>Ensuite</td>
      <td>8939.8232</td>
      <td>5366.6140</td>
      <td>4476.7771</td>
      <td>4476.7771</td>
      <td>Yes</td>
      <td>7.0000</td>
      <td>300000.0000</td>
      <td>1705</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>16658</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1.0000</td>
      <td>1</td>
      <td>nan</td>
      <td>2015-04-30</td>
      <td>nan</td>
      <td>57</td>
      <td>591</td>
      <td>29.0000</td>
      <td>0</td>
      <td>No</td>
      <td>21693.4626</td>
      <td>379.3190</td>
      <td>234.0668</td>
      <td>234.0668</td>
      <td>Yes</td>
      <td>2.0000</td>
      <td>585600.0000</td>
      <td>5209</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>33655</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2.0000</td>
      <td>2+1</td>
      <td>2.0000</td>
      <td>1999-05-12</td>
      <td>nan</td>
      <td>6</td>
      <td>110</td>
      <td>38.0000</td>
      <td>1</td>
      <td>Exclusive</td>
      <td>15433.2876</td>
      <td>3817.2039</td>
      <td>3025.8908</td>
      <td>3025.8908</td>
      <td>Yes</td>
      <td>8.0000</td>
      <td>355000.0000</td>
      <td>305</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>



### 2.5 Building Storeys
Factor building storeys into four ordinal categories: "Low-rise", "Mid-Rise", "High-Rise", and "Very Tall":

    Low-Rise = <6
    Mid-Rise = 6 to 11 storeys (per City of Toronto guidelines)
    High-Rise = 12 to 49 storeys
    Very Tall = 50+


```python
def codify_storeys(x):
    x = float(x)
    if x is np.nan:
        y = np.nan
    elif x < 6:
        y = 0
    elif x < 12:
        y = 1
    elif x < 50:
        y = 2
    elif x >= 50:
        y = 3
    else:
        print("Input is not a number.")
    return y

resales_df['Building_Storeys'] = [codify_storeys(i) for i in resales_df['Building_Storeys']]
```

### 2.6 Building_Units
Without getting too much into the technical details, we'll standardize building_units using a log transformation so that the data is centered around a mean. This will help us overcome the effects of outliers and highlight the relationship between building size (by units) for our model.


```python
# Let's visualize the difference without log and with log.
plt.clf()
sns.distplot(resales_df['Building_Units'])
plt.show()

resales_df['Building_Units log'] = StandardScaler().fit_transform(np.log(resales_df['Building_Units']).values.reshape(-1,1))
sns.distplot(resales_df['Building_Units log'])
plt.show()
```

    C:\Users\Fabienne\Anaconda3\lib\site-packages\matplotlib\axes\_axes.py:6521: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      alternative="'density'", removal="3.1")
    


![png](output_34_1.png)



![png](output_34_2.png)



```python
resales_df['Building_Units'] = resales_df['Building_Units log']
resales_df.drop('Building_Units log', axis =1, inplace=True)
```

### 2.7 Building Registration Date (Building Age)
Here we interpret the registration date of buildings as the building age. While this may not necessarily be the case for all buildings (certain condo developers are notorious for delaying registering despite being completed a while ago), most builders would register upon completing construction as units are not legally allowed to be sold otherwise.

So let's create a **Building_Age** field using the **Building_Reg_Date** field. As mentioned at the beginning of this post, where buildings have no Registration Date, if it has an Expected Completion date, substitute NaN in **Building_Age** with '-1'. 


```python
def building_age(x):
    if x is np.nan:
        return np.nan
    else:
        year = re.findall('[0-9]{4}', x)[0]
        year = int(year)
        return(2018-year)

resales_df['Building_Age'] = [building_age(i) for i in resales_df['Building_Reg_Date']]

# # If Building has expected completion date, put "-1" for Building Age
resales_df.loc[resales_df['Building_Age'].isnull(), 'Building_Age'] = [int(-1) if str(i) != 'nan' else np.nan for i in resales_df.loc[resales_df['Building_Age'].isnull(), 'Building_Completion']]

print("Median age of buildings is {} years. We'll replace {} missing building_ages with this.".format(resales_df['Building_Age'].median(), sum(resales_df['Building_Age'].isnull())))

resales_df.loc[resales_df['Building_Age'].isnull(), 'Building_Age'] = resales_df['Building_Age'].median()

# Drop columns
resales_df.drop(["Building_Completion", "Building_Reg_Date"], axis = 1, inplace = True)
```

    Median age of buildings is 10.0 years. We'll replace 126 missing building_ages with this.
    

### 2.8 Beds, Beds_Description, Den_Description
One way to assure that "Beds" are not incorrectly entered by realtors is to compare them with the number of bathrooms. (e.g if there are Bath == 3 and Beds == 0, it is unlikely that this is a studio, and we must look to Beds_Description.).

We must also do the basic cleaning and replace full words with integers. Studio is 0, 1 Bedroom is 1, 1 Bed+Den is 1.5, etc.


```python
def clean_beds(row):
    baths = row['Bath']
    beds = row['Beds']
    beds_description = row['Beds_Description']
    den_description = row['Den_Description']
    
    if '+' in str(beds):
        new_den = 0.5
    else:
        new_den = 0
    
    if 'Studio' in str(beds):
        new_beds = 0
    elif str(beds) != 'nan':        

        new_beds = re.findall('^[0-9]{1}',beds)[0]
        new_beds = int(new_beds)

        # Test for accuracy of "Beds" field. Use Beds_Description if more baths than beds.
        if str(baths) != 'nan':
            if new_beds == 0 & int(baths) > 1:
                if beds_description is not np.nan:
                    new_beds = beds_description
                else:
                    new_beds = 0
            if not new_den:
                new_den = den_description
        elif str(baths) == 'nan':
            new_beds = int(beds)
    
    elif str(beds) == 'nan':
        new_beds = float(beds_description)
        new_den = float(den_description)
    
    
    try:
        bed_type = new_beds + new_den
    except:
        bed_type = np.nan
        
    return bed_type

bed_type = []
for row in range(len(resales_df)):
    bed_type.append(clean_beds(resales_df.loc[row,:]))

resales_df['Bedrooms'] = bed_type

print('{:,} listings do not have "Beds" values. Clean_Beds function reduced \
this number down to {:,}.'.format(sum(resales_df['Beds'].isnull()), sum(pd.Series(bed_type).isnull())))

# Drop columns
resales_df.drop(['Beds', 'Den_Description', 'Beds_Description'], axis = 1, inplace = True)

# Drop Null Bedrooms
resales_df.dropna(subset = ['Bedrooms'], axis = 0, inplace = True)

# Reset Index
resales_df.reset_index(drop=True, inplace = True)

resales_df.sample(20)
```

    286 listings do not have "Beds" values. Clean_Beds function reduced this number down to 90.
    




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Amenities__Gym / Exercise Room</th>
      <th>Amenities__Party Room</th>
      <th>Amenities__Pool</th>
      <th>Bath</th>
      <th>Building_Storeys</th>
      <th>Building_Units</th>
      <th>Days on Market</th>
      <th>Locker</th>
      <th>Nearest_BRT</th>
      <th>Nearest_GO</th>
      <th>Nearest_HOT</th>
      <th>Nearest_SS</th>
      <th>Parking</th>
      <th>Size_Range</th>
      <th>Sold Price</th>
      <th>Unit</th>
      <th>Balcony_Enclosed</th>
      <th>Balcony_Juliette</th>
      <th>Balcony_None</th>
      <th>Balcony_Open</th>
      <th>Balcony_Terrace</th>
      <th>ajax</th>
      <th>aurora</th>
      <th>brampton</th>
      <th>burlington</th>
      <th>markham</th>
      <th>milton</th>
      <th>mississauga</th>
      <th>newmarket</th>
      <th>oakville</th>
      <th>oshawa</th>
      <th>pickering</th>
      <th>richmondhill</th>
      <th>toronto</th>
      <th>vaughan</th>
      <th>whitby</th>
      <th>Building_Age</th>
      <th>Bedrooms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10776</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2.0000</td>
      <td>2</td>
      <td>0.5202</td>
      <td>26.0000</td>
      <td>Owned</td>
      <td>9825.7275</td>
      <td>2928.2639</td>
      <td>823.1426</td>
      <td>823.1426</td>
      <td>Yes</td>
      <td>5.0000</td>
      <td>415000.0000</td>
      <td>3207</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>14.0000</td>
      <td>2.0000</td>
    </tr>
    <tr>
      <th>13764</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0000</td>
      <td>2</td>
      <td>1.1652</td>
      <td>5.0000</td>
      <td>Owned</td>
      <td>16148.7624</td>
      <td>977.7859</td>
      <td>195.0677</td>
      <td>195.0677</td>
      <td>Yes</td>
      <td>3.0000</td>
      <td>568000.0000</td>
      <td>1210</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>12.0000</td>
      <td>1.5000</td>
    </tr>
    <tr>
      <th>5492</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1.0000</td>
      <td>0</td>
      <td>-1.1069</td>
      <td>35.0000</td>
      <td>Exclusive</td>
      <td>17509.3515</td>
      <td>241.7032</td>
      <td>241.7032</td>
      <td>23755.3748</td>
      <td>Yes</td>
      <td>4.0000</td>
      <td>368000.0000</td>
      <td>212</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>4.0000</td>
      <td>1.5000</td>
    </tr>
    <tr>
      <th>37882</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2.0000</td>
      <td>2</td>
      <td>0.6121</td>
      <td>0.0000</td>
      <td>Owned</td>
      <td>15219.6669</td>
      <td>1482.6786</td>
      <td>56.7803</td>
      <td>56.7803</td>
      <td>Yes</td>
      <td>4.0000</td>
      <td>685000.0000</td>
      <td>Ph1417</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>2.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>25355</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3.0000</td>
      <td>2</td>
      <td>0.1893</td>
      <td>61.0000</td>
      <td>Owned</td>
      <td>22116.9982</td>
      <td>635.0858</td>
      <td>364.0511</td>
      <td>364.0511</td>
      <td>Yes</td>
      <td>13.0000</td>
      <td>2375000.0000</td>
      <td>Ph1204</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>2.0000</td>
      <td>2.5000</td>
    </tr>
    <tr>
      <th>33926</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0000</td>
      <td>1</td>
      <td>-0.2528</td>
      <td>16.0000</td>
      <td>Owned</td>
      <td>12469.6962</td>
      <td>3213.2505</td>
      <td>3213.2505</td>
      <td>4282.8939</td>
      <td>Yes</td>
      <td>3.0000</td>
      <td>427000.0000</td>
      <td>524</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>13.0000</td>
      <td>1.5000</td>
    </tr>
    <tr>
      <th>29512</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2.0000</td>
      <td>2</td>
      <td>0.7641</td>
      <td>2.0000</td>
      <td>No</td>
      <td>8664.9045</td>
      <td>784.8128</td>
      <td>784.8128</td>
      <td>1298.6625</td>
      <td>Yes</td>
      <td>6.0000</td>
      <td>396000.0000</td>
      <td>3024</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>7.0000</td>
      <td>2.5000</td>
    </tr>
    <tr>
      <th>29667</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2.0000</td>
      <td>2</td>
      <td>0.2269</td>
      <td>21.0000</td>
      <td>Owned</td>
      <td>9555.6852</td>
      <td>1435.3865</td>
      <td>284.7194</td>
      <td>284.7194</td>
      <td>Yes</td>
      <td>5.0000</td>
      <td>560000.0000</td>
      <td>NaN</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>6.0000</td>
      <td>2.5000</td>
    </tr>
    <tr>
      <th>22037</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0000</td>
      <td>2</td>
      <td>0.9895</td>
      <td>10.0000</td>
      <td>Owned</td>
      <td>8540.6337</td>
      <td>1645.8164</td>
      <td>292.6288</td>
      <td>292.6288</td>
      <td>Yes</td>
      <td>3.0000</td>
      <td>390000.0000</td>
      <td>307</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>15.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>37887</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0000</td>
      <td>2</td>
      <td>0.6121</td>
      <td>7.0000</td>
      <td>No</td>
      <td>15219.6669</td>
      <td>1482.6786</td>
      <td>56.7803</td>
      <td>56.7803</td>
      <td>No</td>
      <td>2.0000</td>
      <td>488000.0000</td>
      <td>1006</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>2.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>19280</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2.0000</td>
      <td>1</td>
      <td>-1.6169</td>
      <td>8.0000</td>
      <td>Owned</td>
      <td>14622.0906</td>
      <td>6263.4953</td>
      <td>1137.4952</td>
      <td>1137.4952</td>
      <td>Yes</td>
      <td>6.0000</td>
      <td>505000.0000</td>
      <td>1001</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>20.0000</td>
      <td>2.0000</td>
    </tr>
    <tr>
      <th>10189</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0000</td>
      <td>2</td>
      <td>0.7152</td>
      <td>0.0000</td>
      <td>Owned</td>
      <td>521.2839</td>
      <td>2389.3890</td>
      <td>521.2839</td>
      <td>6586.3408</td>
      <td>Yes</td>
      <td>3.0000</td>
      <td>425000.0000</td>
      <td>1207</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>5.0000</td>
      <td>1.5000</td>
    </tr>
    <tr>
      <th>31420</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2.0000</td>
      <td>2</td>
      <td>-0.2658</td>
      <td>0.0000</td>
      <td>Owned</td>
      <td>1855.8033</td>
      <td>1043.7104</td>
      <td>1043.7104</td>
      <td>6858.2224</td>
      <td>Yes</td>
      <td>8.0000</td>
      <td>525000.0000</td>
      <td>1708</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>40.0000</td>
      <td>3.0000</td>
    </tr>
    <tr>
      <th>10242</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0000</td>
      <td>2</td>
      <td>1.3197</td>
      <td>5.0000</td>
      <td>No</td>
      <td>13553.8732</td>
      <td>824.8822</td>
      <td>181.9995</td>
      <td>181.9995</td>
      <td>Yes</td>
      <td>1.0000</td>
      <td>380000.0000</td>
      <td>1920E</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>3.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>20589</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2.0000</td>
      <td>2</td>
      <td>-0.3121</td>
      <td>0.0000</td>
      <td>Owned</td>
      <td>19357.5545</td>
      <td>3031.0464</td>
      <td>150.3403</td>
      <td>150.3403</td>
      <td>Yes</td>
      <td>7.0000</td>
      <td>1000000.0000</td>
      <td>1110</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>10.0000</td>
      <td>2.0000</td>
    </tr>
    <tr>
      <th>38254</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2.0000</td>
      <td>2</td>
      <td>-1.2155</td>
      <td>5.0000</td>
      <td>No</td>
      <td>5990.5706</td>
      <td>1850.7490</td>
      <td>427.0249</td>
      <td>427.0249</td>
      <td>Yes</td>
      <td>8.0000</td>
      <td>907000.0000</td>
      <td>1105</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>17.0000</td>
      <td>2.5000</td>
    </tr>
    <tr>
      <th>12127</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0000</td>
      <td>2</td>
      <td>0.5011</td>
      <td>1.0000</td>
      <td>No</td>
      <td>21536.6623</td>
      <td>861.4119</td>
      <td>13.2212</td>
      <td>13.2212</td>
      <td>No</td>
      <td>1.0000</td>
      <td>482000.0000</td>
      <td>1802</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>7.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>19553</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0000</td>
      <td>2</td>
      <td>-0.3530</td>
      <td>44.0000</td>
      <td>Ensuite</td>
      <td>5573.0430</td>
      <td>2016.5038</td>
      <td>2016.5038</td>
      <td>17271.0410</td>
      <td>Yes</td>
      <td>6.0000</td>
      <td>260000.0000</td>
      <td>514</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>12.0000</td>
      <td>3.0000</td>
    </tr>
    <tr>
      <th>15298</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0000</td>
      <td>2</td>
      <td>0.5650</td>
      <td>21.0000</td>
      <td>No</td>
      <td>22056.2952</td>
      <td>316.8547</td>
      <td>297.8366</td>
      <td>297.8366</td>
      <td>No</td>
      <td>3.0000</td>
      <td>484000.0000</td>
      <td>1808</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>11.0000</td>
      <td>1.5000</td>
    </tr>
    <tr>
      <th>42119</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2.0000</td>
      <td>2</td>
      <td>1.0513</td>
      <td>1.0000</td>
      <td>Owned</td>
      <td>19017.8672</td>
      <td>2823.6699</td>
      <td>400.5852</td>
      <td>400.5852</td>
      <td>Yes</td>
      <td>5.0000</td>
      <td>730000.0000</td>
      <td>815</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>3.0000</td>
      <td>2.0000</td>
    </tr>
  </tbody>
</table>
</div>



At this point, it's worth doing a quick visualization of what our pricing data looks like. 


```python
plt.figure(figsize=(15,8))
ax = sns.boxplot(x='Bedrooms', y='Sold Price', data = resales_df, hue = "toronto")

ax.tick_params(labelsize=14)
ax.set_ylabel(ylabel="Resale Price",fontsize = 20)
ax.set_xlabel(xlabel = "No. of Bedrooms", fontsize = 18)
ax.legend(labels = ["Outside Toronto", "Toronto"], fontsize = 14)

ax.set_ylim([0,2000000])
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('${x:,.0f}'))
plt.title('Bedroom Type vs Condo Apartment Resale Price (Toronto vs. 905)', fontsize=20)
plt.show()
```


![png](output_41_0.png)


### 2.9 Baths

No point standardizing Baths because it is a discrete variable. We can however replace NaN with Median values.


```python
resales_df.loc[resales_df['Bath'].isnull(), 'Bath'] = resales_df['Bath'].median()
```

### 2.10 Days on Market
Clean and standardize.


```python
resales_df['Days on Market'].describe()
```




    count   42675.0000
    mean       14.7788
    std        18.6044
    min         0.0000
    25%         4.0000
    50%         8.0000
    75%        19.0000
    max       399.0000
    Name: Days on Market, dtype: float64




```python
# Clean data where Days on Market is NaN. Replace the values with median.
print("There are {} missing DOM values. We will replace with median, {:,}.".format(sum(resales_df['Days on Market'].isnull()), resales_df['Days on Market'].median()))
resales_df.loc[resales_df['Days on Market'].isnull(), 'Days on Market'] = resales_df['Days on Market'].median()

# Quick exploration of Days on market shows distribution is skewed right by some outliers. 
sns.distplot(resales_df['Days on Market'])
plt.show()

# We will overcome this via standardization.
# Log transformation yields an irregular distribution so we use Square Root Transformation instead. 
resales_df['Days on Market'] = StandardScaler().fit_transform(np.sqrt(resales_df['Days on Market']).values.reshape(-1,1))
sns.distplot(resales_df['Days on Market'])
plt.title("Standardized Days on Market")
plt.show()
```

    There are 2 missing DOM values. We will replace with median, 8.0.
    

    C:\Users\Fabienne\Anaconda3\lib\site-packages\matplotlib\axes\_axes.py:6521: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      alternative="'density'", removal="3.1")
    


![png](output_46_2.png)



![png](output_46_3.png)


### 2.11. Nearest Higher-Order Transit Stop
Let's standardize distances to the nearest Higher-Order Transit stops.


```python
# Let's take a quick look at distribution
plt.figure(1)

plt.subplot(221)
sns.distplot(resales_df['Nearest_SS'])
plt.title('Nearest Subway Station')
plt.grid(True)

plt.subplot(222)
sns.distplot(resales_df['Nearest_HOT'])
plt.title('Nearest HOT')
plt.grid(True)

plt.subplot(223)
sns.distplot(resales_df['Nearest_BRT'])
plt.title('Nearest BRT')
plt.grid(True)

plt.subplot(224)
sns.distplot(resales_df['Nearest_GO'])
plt.title('Nearest GO')
plt.grid(True)

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.8,
                    wspace=0.35)
plt.show()
```

    C:\Users\Fabienne\Anaconda3\lib\site-packages\matplotlib\axes\_axes.py:6521: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      alternative="'density'", removal="3.1")
    


![png](output_48_1.png)



```python
# Visualize whether Square Root or Log is best for standardizing the data
resales_df['Nearest_SS log'] = StandardScaler().fit_transform(np.log(resales_df['Nearest_SS']).values.reshape(-1,1))
resales_df['Nearest_SS sqrt'] = StandardScaler().fit_transform(np.sqrt(resales_df['Nearest_SS']).values.reshape(-1,1))
resales_df['Nearest_HOT log'] = StandardScaler().fit_transform(np.log(resales_df['Nearest_HOT']).values.reshape(-1,1))
resales_df['Nearest_HOT sqrt'] = StandardScaler().fit_transform(np.sqrt(resales_df['Nearest_HOT']).values.reshape(-1,1))
resales_df['Nearest_BRT log'] = StandardScaler().fit_transform(np.log(resales_df['Nearest_BRT']).values.reshape(-1,1))
resales_df['Nearest_BRT sqrt'] = StandardScaler().fit_transform(np.sqrt(resales_df['Nearest_BRT']).values.reshape(-1,1))
resales_df['Nearest_GO log'] = StandardScaler().fit_transform(np.log(resales_df['Nearest_GO']).values.reshape(-1,1))
resales_df['Nearest_GO sqrt'] = StandardScaler().fit_transform(np.sqrt(resales_df['Nearest_GO']).values.reshape(-1,1))

plt.clf()

# Let's take a quick look at distribution
plt.figure(1)

plt.subplot(421)
sns.distplot(resales_df['Nearest_SS log'])
plt.grid(True)
plt.subplot(422)
sns.distplot(resales_df['Nearest_SS sqrt'])
plt.grid(True)

plt.subplot(423)
sns.distplot(resales_df['Nearest_HOT log'])
plt.grid(True)
plt.subplot(424)
sns.distplot(resales_df['Nearest_HOT sqrt'])
plt.grid(True)

plt.subplot(425)
sns.distplot(resales_df['Nearest_BRT log'])
plt.grid(True)
plt.subplot(426)
sns.distplot(resales_df['Nearest_BRT sqrt'])
plt.grid(True)

plt.subplot(427)
sns.distplot(resales_df['Nearest_GO log'])
plt.grid(True)
plt.subplot(428)
sns.distplot(resales_df['Nearest_GO sqrt'])
plt.grid(True)

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=1.8,
                    wspace=0.35)
plt.show()
```

    C:\Users\Fabienne\Anaconda3\lib\site-packages\matplotlib\axes\_axes.py:6521: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      alternative="'density'", removal="3.1")
    


![png](output_49_1.png)


It would appear log transformations offer more consistency to our transformations. We'll use this method.


```python
resales_df['Nearest_GO'] = resales_df['Nearest_GO log']
resales_df['Nearest_BRT'] = resales_df['Nearest_BRT log']
resales_df['Nearest_SS'] = resales_df['Nearest_SS log']
resales_df['Nearest_HOT'] = resales_df['Nearest_HOT log']

resales_df.drop(['Nearest_GO log', 'Nearest_GO sqrt','Nearest_BRT log','Nearest_BRT sqrt',
                       'Nearest_SS log','Nearest_SS sqrt', 'Nearest_HOT log','Nearest_HOT sqrt'], 
                axis = 1, inplace = True)
```

### 2.12. Parking (Binary Encoding)
1 for Yes parking, 0 for no parking.


```python
resales_df['Parking'] = [1 if i == 'Yes' else 0 for i in resales_df['Parking']]
```

### 2.13. Lockers (Binary Encoding)


```python
resales_df['Locker'] = [0 if i == 'No' or i == 'Common' else 1 for i in resales_df['Locker']]
```

### 2.14 Unit (Construct two new binary features: Is_TH, Is_PH)

From our **Units** field, we can infer additional information, such as whether it is a townhouse (while arguably a separate housing type, it's common practice for the real estate industry to include data on townhouses at the base of a condominium apartment in market analyses). We can also parse out of something is a penthouse. We'll create two new fields ("is_TH" and "is_PH") with binary encoding for this.


```python
resales_df['is_TH'] = [1 if i == True else 0 for i in resales_df['Unit'].str.lower().str.contains('th')]
resales_df['is_PH'] = [1 if i == True else 0 for i in resales_df['Unit'].str.lower().str.contains('ph')]

# Drop unit column
resales_df.drop('Unit', axis = 1, inplace=True)
```

And that's it for cleaning and encoding! 

## 3. Modelling

### Split into Training and Testing Data

Now that our data is finally ready, we must split it into data our model can train on, and data for testing. We'll randomly split 80% of it for training and the remainig 20% for testing.


```python
from sklearn.model_selection import train_test_split 

train, test = train_test_split(resales_df, test_size=0.2, random_state=0)
print("Total sample size = {:,}; training sample size = {:,}, testing sample size = {:,}.".format(\
     resales_df.shape[0],train.shape[0],test.shape[0]))
```

    Total sample size = 42,677; training sample size = 34,141, testing sample size = 8,536.
    


```python
df_train = train.loc[:,resales_df.columns]
X_train = df_train.drop(['Sold Price'], axis=1)
y_train = df_train.loc[:, ['Sold Price']]

df_test = test.loc[:,resales_df.columns]
X_test = df_test.drop(['Sold Price'], axis=1)
y_test = df_test.loc[:, ['Sold Price']]
```

### Choosing Our Model
We will test out three basic models to predict our data: Linear Regression, Random Forest, and Gradient Boosting.

### 1.) Linear Regression


```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

# Create the regressor
linreg = LinearRegression()

# Fit the regressor to the training data
linreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = linreg.predict(X_test)

# Compute 5-fold cross-validation scores: cv_scores
cv_scores_linreg = cross_val_score(linreg, X_train, y_train, cv=5)

print("R^2: {:.2f}".format(linreg.score(X_test, y_test)))

mae = abs(y_test - y_pred).mean().item()
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Mean Absolute Error: {:.2f}".format(mae))
print("Root Mean Squared Error: {:.2f}".format(rmse))

print("Average 5-Fold CV Score: {:.5f}".format(np.mean(cv_scores_linreg)))
print(cv_scores_linreg)
```

    R^2: 0.60
    Mean Absolute Error: 112630.24
    Root Mean Squared Error: 185461.19
    Average 5-Fold CV Score: 0.59517
    [0.60293903 0.59172345 0.56290829 0.59449235 0.62379104]
    

### 2.) Random Forest
We will use two forests here. One will be set to the default of 10 decision trees while the other will be set to 100 decision trees. While increasing the number of trees tends to improve the accuracy of our model, we run the risk of overfitting to the training data if we set this number too high. A high number of trees will also slow down the model. However, given the relatively small size of our dataset and number of features, 100 decision trees should be adequate for our predictor.


```python
random_forest = RandomForestRegressor(n_estimators = 10, random_state = 0)

start = time.time()

random_forest.fit(X_train, y_train)

y_pred_rf = random_forest.predict(X_test)

print("Random Forest (10 (default) trees) R^2: {:.4f}".format(random_forest.score(X_test, y_test)))

mae_rf = abs(y_test.values.ravel() - y_pred_rf).mean().item()
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print("Mean Absolute Error: {:.2f}".format(mae_rf))
print("Root Mean Squared Error: {:.2f}".format(rmse_rf))

# Print the 5-fold cross-validation scores
cv_scores_rf = cross_val_score(random_forest, X_train, y_train.values.ravel(), cv=5)
print("Average 5-Fold CV Score: {:.4f}".format(np.mean(cv_scores_rf)))
print(cv_scores_rf)

print("\nTime elapsed: {:.2f} seconds".format(time.time() - start))
```

    C:\Users\Fabienne\Anaconda3\lib\site-packages\ipykernel_launcher.py:5: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      """
    

    Random Forest (10 (default) trees) R^2: 0.8947
    Mean Absolute Error: 50019.37
    Root Mean Squared Error: 95154.96
    Average 5-Fold CV Score: 0.8389
    [0.86282212 0.78101956 0.80838234 0.866611   0.87561959]
    
    Time elapsed: 13.47 seconds
    


```python
random_forest_100 = RandomForestRegressor(n_estimators = 100, random_state = 0)

start = time.time()
random_forest_100.fit(X_train, y_train.values.ravel())

y_pred_rf100 = random_forest_100.predict(X_test)

print("Random Forest (100 trees) R^2: {:.4f}".format(random_forest_100.score(X_test, y_test)))

mae_rf100 = abs(y_test.values.ravel() - y_pred_rf100).mean()
rmse_rf100 = np.sqrt(mean_squared_error(y_test, y_pred_rf100))
print("Mean Absolute Error: {:.2f}".format(mae_rf100))
print("Root Mean Squared Error: {:.2f}".format(rmse_rf100))

cv_scores_rf100 = cross_val_score(random_forest_100, X_train, y_train.values.ravel(), cv=5)
print("Average 5-Fold CV Score: {:.4f}".format(np.mean(cv_scores_rf100)))

print("\nTime elapsed: {:.1f} seconds".format(time.time() - start))
```

    Random Forest (100 trees) R^2: 0.9029
    Mean Absolute Error: 48114.75
    Root Mean Squared Error: 91389.67
    Average 5-Fold CV Score: 0.8536
    
    Time elapsed: 146.3 seconds
    

### 3.) Gradient Boosting
Gradient Boosting is an ensemble method, meaning it is built using many weak learners (i.e. shallow decision trees). Unlike random forest, which builds trees on randomly selected features (thus the name), gradient boosting regressors is an iterative process where each tree is drawn depending on residual errors from the previous tree. There's a fantastic 2-minute explanation [here](https://www.youtube.com/watch?v=GM3CDQfQ4sw). This learning process tends to make Gradient Boosting a method favoured by many in the data science community.

Let's see how well it applies to our model...


```python
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor

start = time.time()

# R squared
gradientboost = ensemble.GradientBoostingRegressor()
gradientboost.fit(X_train, y_train.values.ravel()) 
print('Gradient Boosting R^2: {:.4f}'.format(gradientboost.score(X_test, y_test)))

# MAE and RMSE
y_pred_gb = gradientboost.predict(X_test)
gradientboost_mae = abs(y_pred_gb - y_test.values.ravel()).mean()
gradientboost_rmse = np.sqrt(mean_squared_error(y_pred_gb, y_test.values.ravel()))
print("Mean Absolute Error: {:.2f}".format(gradientboost_mae))
print('Gradient Boosting RMSE: {:.4f}'.format(gradientboost_rmse))

# 5-Fold cross-validation Score
cv_scores_gb = cross_val_score(gradientboost, X_train, y_train.values.ravel(), cv=5)
print("Average 5-Fold CV Score: {:.4f}".format(np.mean(cv_scores_gb)))
print("\nTime elapsed: {:.1f} seconds".format(time.time() - start))
```

    Gradient Boosting R^2: 0.8472
    Mean Absolute Error: 68268.35
    Gradient Boosting RMSE: 114649.2841
    Average 5-Fold CV Score: 0.8053
    
    Time elapsed: 22.5 seconds
    


```python
R2_ = [linreg.score(X_test,y_test), random_forest.score(X_test,y_test), random_forest_100.score(X_test, y_test), gradientboost.score(X_test, y_test)] 
MAE_ = ["${:,.0f}".format(x) for x in [mae, mae_rf, mae_rf100, gradientboost_mae]]
RMSE_ = ["${:,.0f}".format(x) for x in [rmse, rmse_rf, rmse_rf100, gradientboost_rmse]]
cv_5 = [np.mean(x) for x in [cv_scores_linreg, cv_scores_rf, cv_scores_rf100, cv_scores_gb]]

pd.DataFrame(index=['R2', 'Mean Absolute Error','Root Mean Squared Error','Avg 5-Fold Cross-Validation Score'], 
             columns = ["Linear Regr.", "Random Forest", "Random Forest 100", "Gradient Boost"], data = [R2_, MAE_, RMSE_, cv_5])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Linear Regr.</th>
      <th>Random Forest</th>
      <th>Random Forest 100</th>
      <th>Gradient Boost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>R2</th>
      <td>0.6000</td>
      <td>0.8947</td>
      <td>0.9029</td>
      <td>0.8472</td>
    </tr>
    <tr>
      <th>Mean Absolute Error</th>
      <td>$112,630</td>
      <td>$50,019</td>
      <td>$48,115</td>
      <td>$68,268</td>
    </tr>
    <tr>
      <th>Root Mean Squared Error</th>
      <td>$185,461</td>
      <td>$95,155</td>
      <td>$91,390</td>
      <td>$114,649</td>
    </tr>
    <tr>
      <th>Avg 5-Fold Cross-Validation Score</th>
      <td>0.5952</td>
      <td>0.8389</td>
      <td>0.8536</td>
      <td>0.8053</td>
    </tr>
  </tbody>
</table>
</div>



A side-by-side comparison of how the models perform shows that Random Forest set to 100 trees has the least margin of error for predicting prices.

### Improving Our Model
Feature importance is crucial to model optimization. Without getting lost in the technical details, __feature importance__ is a measure of how relevant the independent features are to explaining/predicting our target variable. In a nutshell, they measure the reduction in error in including the feature. 

Below, are the feature importances, per our three models, ranked by importance to our 'Random Forest with 100 Trees' model.


```python
GB_results = pd.DataFrame(columns=['Features','GB Importance'], index = X_train.columns, data = list(zip(X_train.columns, gradientboost.feature_importances_)))
RF_results = pd.DataFrame(columns=['Features','RF Importance'], index = X_train.columns, data = list(zip(X_train.columns,random_forest.feature_importances_)))
RF100_results = pd.DataFrame(columns=['Features', 'RF100 Importance'], index = X_train.columns, data = list(zip(X_train.columns,random_forest_100.feature_importances_)))

importance_df = GB_results.drop('Features', axis=1).join(RF_results.drop('Features', axis=1), how='outer')
importance_df = importance_df.join(RF100_results.drop('Features', axis=1), how='outer')
importance_df.sort_values('RF100 Importance', ascending= False)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GB Importance</th>
      <th>RF Importance</th>
      <th>RF100 Importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Size_Range</th>
      <td>0.3378</td>
      <td>0.5132</td>
      <td>0.5080</td>
    </tr>
    <tr>
      <th>Nearest_SS</th>
      <td>0.0764</td>
      <td>0.1211</td>
      <td>0.1214</td>
    </tr>
    <tr>
      <th>Nearest_BRT</th>
      <td>0.1458</td>
      <td>0.1155</td>
      <td>0.1167</td>
    </tr>
    <tr>
      <th>Building_Age</th>
      <td>0.1173</td>
      <td>0.0657</td>
      <td>0.0671</td>
    </tr>
    <tr>
      <th>Bath</th>
      <td>0.0546</td>
      <td>0.0518</td>
      <td>0.0449</td>
    </tr>
    <tr>
      <th>Nearest_GO</th>
      <td>0.0483</td>
      <td>0.0314</td>
      <td>0.0316</td>
    </tr>
    <tr>
      <th>Days on Market</th>
      <td>0.0165</td>
      <td>0.0198</td>
      <td>0.0253</td>
    </tr>
    <tr>
      <th>Building_Units</th>
      <td>0.0354</td>
      <td>0.0213</td>
      <td>0.0203</td>
    </tr>
    <tr>
      <th>Bedrooms</th>
      <td>0.0321</td>
      <td>0.0135</td>
      <td>0.0159</td>
    </tr>
    <tr>
      <th>Nearest_HOT</th>
      <td>0.0189</td>
      <td>0.0133</td>
      <td>0.0146</td>
    </tr>
    <tr>
      <th>Building_Storeys</th>
      <td>0.0211</td>
      <td>0.0051</td>
      <td>0.0056</td>
    </tr>
    <tr>
      <th>Locker</th>
      <td>0.0102</td>
      <td>0.0032</td>
      <td>0.0036</td>
    </tr>
    <tr>
      <th>Balcony_Open</th>
      <td>0.0015</td>
      <td>0.0035</td>
      <td>0.0029</td>
    </tr>
    <tr>
      <th>Amenities__Party Room</th>
      <td>0.0146</td>
      <td>0.0028</td>
      <td>0.0028</td>
    </tr>
    <tr>
      <th>Balcony_Terrace</th>
      <td>0.0096</td>
      <td>0.0018</td>
      <td>0.0027</td>
    </tr>
    <tr>
      <th>Amenities__Pool</th>
      <td>0.0036</td>
      <td>0.0022</td>
      <td>0.0024</td>
    </tr>
    <tr>
      <th>is_PH</th>
      <td>0.0023</td>
      <td>0.0023</td>
      <td>0.0020</td>
    </tr>
    <tr>
      <th>Balcony_None</th>
      <td>0.0033</td>
      <td>0.0024</td>
      <td>0.0020</td>
    </tr>
    <tr>
      <th>Parking</th>
      <td>0.0103</td>
      <td>0.0017</td>
      <td>0.0018</td>
    </tr>
    <tr>
      <th>Amenities__Gym / Exercise Room</th>
      <td>0.0008</td>
      <td>0.0012</td>
      <td>0.0014</td>
    </tr>
    <tr>
      <th>oakville</th>
      <td>0.0035</td>
      <td>0.0009</td>
      <td>0.0013</td>
    </tr>
    <tr>
      <th>vaughan</th>
      <td>0.0050</td>
      <td>0.0011</td>
      <td>0.0009</td>
    </tr>
    <tr>
      <th>mississauga</th>
      <td>0.0005</td>
      <td>0.0009</td>
      <td>0.0008</td>
    </tr>
    <tr>
      <th>toronto</th>
      <td>0.0015</td>
      <td>0.0009</td>
      <td>0.0008</td>
    </tr>
    <tr>
      <th>markham</th>
      <td>0.0081</td>
      <td>0.0005</td>
      <td>0.0007</td>
    </tr>
    <tr>
      <th>is_TH</th>
      <td>0.0116</td>
      <td>0.0008</td>
      <td>0.0005</td>
    </tr>
    <tr>
      <th>brampton</th>
      <td>0.0000</td>
      <td>0.0005</td>
      <td>0.0003</td>
    </tr>
    <tr>
      <th>richmondhill</th>
      <td>0.0031</td>
      <td>0.0003</td>
      <td>0.0003</td>
    </tr>
    <tr>
      <th>Balcony_Juliette</th>
      <td>0.0000</td>
      <td>0.0004</td>
      <td>0.0003</td>
    </tr>
    <tr>
      <th>Balcony_Enclosed</th>
      <td>0.0000</td>
      <td>0.0003</td>
      <td>0.0003</td>
    </tr>
    <tr>
      <th>burlington</th>
      <td>0.0063</td>
      <td>0.0005</td>
      <td>0.0001</td>
    </tr>
    <tr>
      <th>aurora</th>
      <td>0.0000</td>
      <td>0.0001</td>
      <td>0.0001</td>
    </tr>
    <tr>
      <th>pickering</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>newmarket</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>milton</th>
      <td>0.0000</td>
      <td>0.0001</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>ajax</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>whitby</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>oshawa</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>



We can see from our Feature Importance table that unit size (Size Range) has the greatest impact on predicting price (0.51), followed by distance to Subway/Streetcar/BRT stop (0.12) and Building Age (0.07). Conversely, our model considers whether a listing is in Ajax, Oshawa, or Whitby as having virtually no importance on predicting price. Taking a step back, this is not to say municipality location is unimportant, per se - ceterus parabus, a condo listing in Oshawa would not be priced the same as if it were in Oakville. However, for our model, it has less influence in predicting price in our data, considering other features.

If we wanted to improve this model, one avenue would be to remove features with less importance. Perhaps we'll do that in a follow up post.

### Modelling Summary
We have tried three models: Linear Regression, Random Forest, and Gradient Boosting. Between the three and in terms of R Squared and RMSE, Random Forest performs the best for predicting our housing prices. 

Gradient Boosting is often preferred over Random Forest because of its iterative learning process (adjusting according to the errors of its predecessing models), however, it is prone to overfitting when data has lots of noise. This is the case with our data, scraped raw from a listings site and often with outliers (e.g. luxury units, problematic units, negotiation nuances). As such, Random Forest is a more accurate predictive model for this data.

Let's visualize how our chosen model, Random Forest with 100 trees, performs with its predicted target Y variables vs. actual target Y variables.


```python
fig,ax = plt.subplots()
plt.scatter(x = range(len(y_pred_rf100)), y = y_pred_rf100, alpha = 0.6,  s = 3, c='b', label='Prediction')
plt.scatter(x = range(len(y_pred_rf100)), y = y_test.values, alpha = 0.5, s = 3, c='r', label='Actual')
plt.legend(fontsize=12, markerscale=2)
plt.ylim(0,2000000)
plt.title("Random Forest (100 Trees) Predictions (Blue) vs. Actual Values (Red)", fontsize = 14)
plt.ylabel("Error in Predicted Sold Price")
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('${x:,.0f}'))
plt.xlabel("Observation Number")
plt.show()
```


![png](output_76_0.png)


## 4. Conclusion

With an R Squared value of 0.90, our chosen model of 100 trees typically predicts real estate prices within roughly $48,000 of the actual price - not bad at all! The next step would be to go into production and deploy this on a server...

But before we get too excited, it's important to point out the dynamic nature of the real estate market which would impact the real-life performance of our model. Out of curiosity, I quickly ran the model against roughly 3,800 active listings (on market as of October 31st 2018) and yielded the following results:

![title](2_Analysis/Model against Active.png)

Evidently, this model has a far lower success rate on more recent listings. Perhaps our training data was too old. Perhaps our predictor variables were not as accurate. There are a number of factors the inaccuracy could be due to and they all highlight the quickly evolving and nuanced nature of real estate demand and supply. But I'd argue figuring out the different variables and their impacts on the market is the fun part.

## Closing Points

Moving forward, we're seeing more datapoints permeate the real estate and real estate-adjacent industries, such as exact unit square footage (as opposed to just ranges) and [elevator inspection data](https://www.tssa.org/en/elevating-devices/elevating-devices---open-data.aspx) - which we can use to extrapolate other information about buildings and engineer more features from. Addtionally, we could build and join a database that rates developers and their building track record -- we know certain developers like Tridel are superior in their quality of building materials while others couldn't care less if their floorboards start peeling before building registration. We could also maybe calculate school ratings and distance to the closest school (although this would probably be more useful for predicting single-detached home prices). The possibilities are endless.

While our model works quite well, with more fine-tuning and feature-engineering, I am sure we can build a much more robust model that "understands" the real estate market much better. 

As usual, I'm happy to chat about ideas. Feel free to reach me on [LinkedIn](https://www.linkedin.com/in/fabiennechan/) or [Twitter](http://www.twitter.com/fabiennechan).
