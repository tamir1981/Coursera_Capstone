#!/usr/bin/env python
# coding: utf-8

# ## Tamir Final Coursera Project - Hummus in LONDON
# 

# In[206]:


import numpy as np # library to handle data in a vectorized manner
import pandas as pd 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import json
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors
from sklearn.cluster import KMeans
import folium # map rendering library

print('Libraries imported.')


# The purpose of this work is to simulate a business situation of choosing a location for a restaurant. In this case, the potential developer wants to find a place for a Hummus restaurant (Chickpeas). While asumming that a good location is around main tourist attractions because of the tourist traffic, I will extract a list of the most popular sites in LONDON and use the  Forcesquare site location data to find the ideal location for the restaurant according to the competitors locations and their concentrations. Competitors in this case will be a list of specific category of restaurants who usually compete with Hummus places like: middle-eastern, lebanese, israeli and such.

# Let us assume for simplicity that it is worthwhile to open the restaurant near tourist sites in central London. We will harvest the list of popular tourist sites from this website:

# In[207]:


import requests
import lxml
from bs4 import BeautifulSoup

# Obtaining data London Tourist website:
source = requests.get('https://www.londoncitybreak.com/areas').text
soup = BeautifulSoup(source, 'lxml')
soup.prettify


# In[208]:


#finding the best Atrractions according to the website:
PlaceName = []
td = soup.find_all('a', {"class":"o-page-nav__sub__element__link"})
for line in td[9:67]:
    place = line.get('title')
    PlaceName.append(place)

print(PlaceName)


# In[209]:


# Form a dataframe:
dict = {'Place' : PlaceName}
        
info = pd.DataFrame.from_dict(dict)
info.head()


# In[210]:


#Cleaning the DataFrame from Website garbage:
to_drop = ['Ver todo', 'Monuments and Tourist attractions','Museums and Galleries']
info = info[~info['Place'].isin(to_drop)]


# In[211]:


info


# In[212]:


# Extracting Loction Data for our Tourist Atracctions:
import geocoder


# In[213]:


# Loop the gather the latitude and longitude:
for i, row in info.iterrows():
        # Accessing single Value:
        PlaceName = info.at[i, 'Place']
        
        #retreiving the Data:
        lat_lng = geocoder.arcgis('{}, London, UK'.format(PlaceName))
        lat_lng_coords = lat_lng.latlng
        latitude = lat_lng_coords[0]
        longitude = lat_lng_coords[1]
        
        #Pasting back to the df:
        info.at[i, 'Latitude'] = latitude
        info.at[i, 'Longitude'] = longitude
        
        #print for Fun!!


# In[214]:


info


# In[378]:


#Checking DF size:
print('The dataframe has {} Places'.format(len(info['Place'].unique()))) 


# In[429]:


#Finding london's coordinations:
address = 'London, UK'

geolocator = Nominatim(user_agent="London_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of London are {}, {}.'.format(latitude, longitude))

#### Create a map of London with the places superimposed on top. ####
# In[657]:


# create map of Toronto latitude and longitude values
map_London = folium.Map(width=1000,height=1000,location=[latitude, longitude], zoom_start=10)


# In[658]:


type(map_London)


# In[659]:


# add markers to map
for lat, lng, Place in zip(info['Latitude'], info['Longitude'], info['Place']):
    label = '{}'.format(Place)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=True).add_to(map_London)  

map_London


# In[122]:


#utilizing the Foursquare API to explore the Places and explore them:


# In[123]:


#### Define Foursquare Credentials and Version ####


# In[1]:


CLIENT_ID = 'Put your own' # your Foursquare ID
CLIENT_SECRET = 'Put your own' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[125]:



##Get the neighborhood's latitude and longitude values.##
Place_latitude = info.loc[0, 'Latitude'] # neighborhood latitude value
Place_longitude =info.loc[0, 'Longitude'] # neighborhood longitude value

Place_name = info.loc[0, 'Place'] # neighborhood name

print('Latitude and longitude values of {} are {}, {}.'.format(Place_name, 
                                                               Place_latitude, 
                                                               Place_longitude))


# In[126]:


# Now, let's get the top 100 venues that are in The Beachesl within a radius of 500 meters.#


# In[316]:


radius = 750
LIMIT = 100
# type your answer here
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    Place_latitude, 
    Place_longitude, 
    radius, 
    LIMIT)
url # display URL


# In[317]:


#Send the GET request and examine the resutls


# In[318]:


results = requests.get(url).json()


# In[319]:


results


# In[320]:


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[321]:


#Now we are ready to clean the json and structure it into a *pandas* dataframe.


# In[322]:


venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()


# In[134]:


#And how many venues were returned by Foursquare?


# In[323]:


print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))


# <a id='item2'></a>

# In[324]:


# Let's create a function to repeat the same process to all the neighborhoods in London #


# In[325]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Place', 
                  'Place Latitude', 
                  'Place Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[326]:


#Creating London Places:


# In[327]:


London_venues = getNearbyVenues(names=info['Place'],
                                   latitudes=info['Latitude'],
                                   longitudes=info['Longitude']
                                  )


# In[140]:


#### Let's check the size of the resulting dataframe


# In[465]:


print(London_venues.shape)
London_venues


# In[371]:


# print(London_venues['Venue Category'].unique())
London_venues[London_venues['Venue Category'].str.contains("Fast")]


# In[142]:


#Selecting Only the potential Competitors - Middle Easters Cusine:


# In[469]:


London_Competitiors = London_venues[London_venues['Venue Category'].isin(['Turkish Restaurant','Mediterranean Restaurant',
                                                                           'Israeli Restaurant','Middle Eastern Restaurant',
                                                                           'Kebab Restaurant','Falafel Restaurant',
                                                                           'Halal Restaurant','Iraqi Restaurant', 
                                                                           'Lebanese Restaurant','Persian Restaurant',
                                                                           'Greek Restaurant',
                                                                           'Moroccan Restaurant','Fast Food Restaurant',
                                                                           'Vegetarian / Vegan Restaurant',' Sandwich Place',
                                                                           'Fast Food Restaurant'])]


# In[470]:


London_Competitiors.reset_index(drop=True,inplace=True)


# In[471]:


London_Competitiors.shape


# In[660]:


map_competition = map_London


# In[661]:


map_competition


# In[662]:


# add markers of Competitors on Map along side Places:
for lat, lng, Venue, Type in zip(London_Competitiors['Venue Latitude'], London_Competitiors['Venue Longitude'], London_Competitiors['Venue'], London_Competitiors['Venue Category']):
    label = '{}, {}'.format(Venue, Type)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='pink',
        fill=True,
        fill_color='#ff6666',
        fill_opacity=0.7,
        parse_html=True).add_to(map_competition)  
    
map_competition


# In[336]:


# Let's add HeatMap to increase visibilty of competition:


# In[458]:


from folium import FeatureGroup, LayerControl, Map, Marker
from folium.plugins import HeatMap


# In[462]:


London_Heat = HeatMap( list(zip(London_Competitiors['Venue Latitude'],London_Competitiors['Venue Longitude'])),
                     min_opacity=0.7,
                     radius=25, blur=15,
                     max_zoom=30
                 )


# In[457]:


map_heat = map_London


# In[463]:


# London_Heat.add_to(map_competition)
London_Heat.add_to(map_heat)


# In[464]:


map_heat


# In[477]:


# Let's find out how many unique categories can be curated from all the returned venues
print('There are {} uniques categories.'.format(len(London_Competitiors['Venue Category'].unique())))


# In[478]:


## Analyze Each Neighborhood


# In[684]:


# one hot encoding
London_onehot = pd.get_dummies(London_Competitiors[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
London_onehot['Place'] = London_Competitiors['Place'] 

# move neighborhood column to the first column
fixed_columns = [London_onehot.columns[-1]] + list(London_onehot.columns[:-1])
London_onehot = London_onehot[fixed_columns]

London_onehot


# In[480]:


#And let's examine the new dataframe size.


# In[481]:


# Sorting the Hardest Areas with the most competitors:

London_Totals = London_onehot.groupby(['Place']).sum()


# In[482]:


London_Totals['Total'] = London_Totals.sum(axis=1)


# In[683]:


London_Totals.sort_values(by='Total', ascending=False, inplace=True)
London_Totals


# In[499]:


London_Totals.to_excel('table.xlsx')


# In[564]:


totals = London_Totals[["Total"]]
totals.head()


# In[565]:


# Visualizing Places according to competitors:

import matplotlib as mpl
import matplotlib.pyplot as plt


# In[637]:


x = totals.plot(kind='barh',
                figsize=(8, 10),
                color= 'blue',
                 zorder=1,
                 width=0.85,
                 title='Total Competitors By Location', 
               )
plt.savefig("myplot.png", dpi = 400)


# In[290]:


# Next, let's group rows by Places and by taking the mean of the frequency of occurrence of each category


# In[638]:


London_grouped = '' 


# In[639]:


London_grouped = London_onehot.groupby('Place').mean().reset_index()


# In[640]:


London_grouped.head()


# In[641]:


# Let's confirm the new size


# In[642]:


London_grouped.shape


# In[643]:


# Let's print each Place along with the top 5 most common venues


# In[644]:


num_top_venues = 5

for attraction in London_grouped['Place']:
    print("----"+attraction+"----")
    temp = London_grouped[London_grouped['Place'] == attraction].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[297]:


# Putting the DATA into df:


# In[645]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[646]:


#Now let's create the new dataframe and display the top 10 venues for each Place.


# In[647]:


Places_venues_sorted=''


# In[648]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Place']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
Places_venues_sorted = pd.DataFrame(columns=columns)
Places_venues_sorted['Place'] = London_grouped['Place']

for ind in np.arange(London_grouped.shape[0]):
    Places_venues_sorted.iloc[ind, 1:] = return_most_common_venues(London_grouped.iloc[ind, :], num_top_venues)

Places_venues_sorted


# In[357]:


## Cluster Places:


# In[358]:


# Run *k*-means to cluster the Places into 5 clusters:


# In[649]:


# set number of clusters
kclusters = 5

London_grouped_clustering = London_grouped.drop('Place', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(London_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# In[650]:


# add clustering labels
Places_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

London_merged = info

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
London_merged = London_merged.join(Places_venues_sorted.set_index('Place'), on='Place')

London_merged.head() # check the last columns!


# In[651]:


London = London_merged.dropna(inplace=True)


# In[ ]:


#Finally, let's visualize the resulting clusters: 
#We avoid the "NaN" values in DF:


# In[653]:


London_merged


# In[664]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(London_merged['Latitude'], London_merged['Longitude'], London_merged['Place'], London_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    if cluster != 'nan':
#         if cluster != 0:
        try:
            cluster = int(cluster)
            folium.CircleMarker(
            [lat, lon],
            radius=8,
            popup=label,
            color=rainbow[cluster-1],
            fill=True,
            fill_color=rainbow[cluster-1],
            fill_opacity=0.7).add_to(map_competition)
        except:
            continue
map_competition


# In[673]:


combine = pd.merge(London_Totals, London_merged, on="Place")


# In[678]:


combine


# In[681]:


combine.drop_duplicates(subset='Place',inplace=True)


# In[682]:


# Let's analize the meaning of the Clustering: !!!!![See summery in Presantation]!!!!
combine.sort_values(by=['Cluster Labels','Total'],ascending=False)


# In[ ]:




