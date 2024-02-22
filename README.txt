Description: 
The question that I am interested in answering for my project is weather there is a correlation between weather and crime rate where I'd expect that as the temperature increases, the number of crimes should also increase. This is based off of the heat hypothesis which mentions that hotter temperatures would prompt more aggressive behaviors and thus an increase in crime. I'm also interested in seeing whether less crime is observed on rainy days versus sunny days as I expect less foot traffic outside during rainy days. To answer these questions, I have decided to use three data sources, 2 for weather and 1 that contains the crime reports in San Francisco between the range of January 2019 to January 2021. There will be four datasets in the form of a csv file from three data sources in this code with one combined file: 

1. incidents.csv
2. noaa.csv
3. weather.csv
4. combined.csv

They will be located in folder titled "data" in the zipfile. 

The analysis results & supporting visualization png's will be located in the "analysis" folder. 

Requirements: The following requirements need to be satisfied in order to run this code successfully: 

1. Python 3.9+
2. [pipenv] (https://pipenv.pypa.io/en/latest/)
3. Google Chrome (https://www.google.com/chrome/?brand=CHBD&gclid=CjwKCAjw9J2iBhBPEiwAErwpeWxkC6Z5mSzZhmIdamySxRKE5WqzBd20wW2ESvGv-g9yuqAd9Vv1MRoCZzAQAvD_BwE&gclsrc=aw.ds)
4. Terminal (with ANSI Color Support)
5. Chrome driver

Packages to be installed: 
1. webdriver-manager
2. pandas
3. requests
4. beautifulsoup4
5. tabulate
6. rich
7. pyyaml
8. selenium
9. seaborn
10. matplotlib
11. scipy

To install the packages above, use the following command: pip/pip3 install package name (e.g., pip install pandas)
You can also use the command "pip install -r requirements.txt" or "pip3 install -r requirements.txt" to install the necessary packages and libraries to run this code. 

Data sources in depth: 
1. San Francisco Police Department Incident Reports ranging from 2018 to present: (https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-2018-to-Present/wg3w-h783)

The URL is already embedded in the code and thus it does not need to be given as input by the user. Using the requests library, I have accessed the json from the URL and created a file named incidents.csv. This dataset consists of 276,940 rows and 27 columns and is scraped in 5000 row increments. To name a few important columns in the dataset: 
1. Incident date
2. Incident year
3. Incident category (e.g., Lost property, Fraud, Stolen property, Burglary etc)
4. Police district 

From this dataset, I have used the incident date and year in my final analysis. 

2. Historical San Francisco Weather Data Daily Summaries (2019 to 2021): (https://www.ncdc.noaa.gov/cdo-web/datasets/GHCND/stations/GHCND:USW00023272/detail)

This dataset was exported from the NOAA website as a csv file (noaa.csv) and it contains 736 rows with 8 columns. The columns in this dataset are: 
1. Station (which weather station the weather data is collected from)
2. Name (San Francisco Downtown Station)
3. Date
4. Precipitation
5. Rain (yes/no)
6. TAVG (Average temperature)
7. TMAX (Maximum temperature)
8. TMIN (Minimum temperature)

3. Historical San Francisco Weather Data Daily Summaries (2019 to 2021): (https://www.wunderground.com/history/monthly/us/ca/san-bruno/KSFO/date/2019-1)

The link above shows the historical weather data for January 2019, there is a tab under the monthly section to view each month individually in a dynamic table on the website. The link does not need to be provided by the user as input as it is already included in the code with the parameters set to scrape from 1/1/2019 to 1/31/2021. As the tabular data on the website was a dynamic table, beautifulsoup wasn't able to fetch the data so I used selenium with chrome driver to get the HTML of the table. Once that was obtained, it was passed off to beautifulsoup to start the scraping of the tables and saved as a csv file titled (weather.csv). There are 761 rows and 17 columns in the dataset such as: 
1. Date
2. temp_max (maximum temperature)
3. temp_avg (average temperature)
4. temp_min (minimum temperature)
5. precipitation_total (0 meaning no rain and > 0 indicating rain on the day)

4. Combined.csv (Final merged dataset used for analysis)

This dataset was made by merging certain columns from each of the datasets mentioned above into one csv file and the key column between all the files is the date. 
Columns in this file:
   - Date (Each dataset has a date column with the range starting from 1/1/2019 to 1/31/2021)
   - Rain (Taken from the noaa dataset)
   - Average Temp (NOAA) (Taken from the noaa dataset)
   - Average Temp (Wunderland) (Taken from the weather dataset)
   - Number of Crimes (Taken from the incidents dataset)
   - Average Temp (Combined) (New column taken by averaging out the two average temperature values)

Running the code: 
This zipfile contains one Python script, "main.py" and it can be run in four different modes: default, scrape, static and analyze mode depending on what you would like it to do.  

Default mode: To run the code in default mode, type the command - python3 main.py
In this mode, two datasets ("incidents.csv" & "weather.csv") will be scraped and fetched from the API specifically. 
It will also open the noaa.csv incidents file and the time taken to run this code varies between 3-5 minutes and there will be a timer printed in the output to show exactly how many seconds it takes to run the code. 
Once the data is scraped/fetched/loaded and saved, a user input statement will be generated to ask the user if they would like to combine the datasets to create (combined.csv). If user answers yes (y), it will proceed to combine the dataset. 
The next user input statement will then be generated to ask the user if they would like to run the analysis on the combined dataset. If user answer yes (y), it will proceed to do the analysis and print out the results in the console. 
If user answers no (n) for the first input statement, the code will break and stop running once all the data has been scraped/fetched/loaded.Please see the additional print statements in the generated output once the code is run to see exactly what is happening. 

Scrape mode: To run the code in scrape mode, type the command - python3 main.py --scrape
The expected output in this mode is a small sample of the scraped data & data gathered from the API displayed in the form of a table. It should print the column headers along with 10 rows of data and the total row and columns of the dataset. The total time taken in seconds will also be printed on the bottom. 

Static mode: To run the code in static mode, type the commands: 
- python3 main.py --static ./data/noaa.csv
- python3 main.py --static ./data/weather.csv
- python3 main.py --static ./data/incidents.csv
- python3 main.py --static ./data/combined.csv

In this mode, using the commands above, it will print out the column header, 10 rows of data, total rows and columns of the dataset along with the time taken to run the code. There are four separate commands to display the four different datasets. 

Analyze mode: To run the code in analyze mode, type the command - python3 main.py --analyze
The expected output in this mode is the results of the analysis (t-test, Pearson, Spearman's and Kendall's Correlation) between Average Temperature and Crime Rate as well as Rain and Crime Rate. 
The supporting visualizations (scatterplot and bar graph will also be generated and saved). The results of each test will be saved in a .txt file and the visualizations in a .png file.

You can use python3 main.py --help to see the additional available arguments but they will all be run with a default start and end date as specified in the description of the project. 

Extensibility & Architecture of Code: 

This Python script has several components and is used for scraping and fetching specific datasets.
It is designed to fetch data from the Weather Underground and San Francisco Police Department (SFPD) incident datasets.
The script provides several functionalities, such as loading and saving datasets from/to different file formats,
displaying datasets in a table format, and fetching data from APIs or by scraping websites. This script also has the capability to create a new dataset
by taking the information from specific columns and writing it into a new file. Correlational analysis will then be performed on the file and supporting visualizations
will be generated and saved. 

Here's an overview of the script's components:

1. Importing necessary libraries: The script imports necessary libraries such as argparse, json, os, time, bs4 (
   BeautifulSoup4), pandas, requests, yaml, rich, selenium, and tabulate.
2. Defining utility functions: Several utility functions are defined, such as `snake_to_normal_case`, `get_file_type`,
   and `chrome_get_htmls`, which are used throughout the script.
3. Dataset classes: The script contains multiple classes for handling datasets. The base class, `Dataset`, provides
   common functionality for loading, saving, and displaying datasets. Derived classes
   include `NOAAWeatherDataset`, `WUGWeatherDataset`, and `IncidentDataset`, which provide additional functionality
   specific to their respective data sources.
4. `NOAAWeatherDataset`: This class represents the NOAA weather dataset and inherits from the `Dataset` class.
5. `WUGWeatherDataset`: This class represents the Weather Underground weather dataset and inherits from the `Dataset`
   class. It provides a `fetch_data` method for scraping data from the Weather Underground website.
6. `IncidentDataset`: This class represents the San Francisco Police Department incident dataset and inherits from
   the `Dataset` class. It provides a `fetch_data` method for fetching data from the SFPD API.
7. `CombinedDataset`: This class represents the combined dataset and inherits from the `Dataset` class. It provides
   methods for combining other 3 datasets, analyzing the combined dataset, and visualizing the results.
8. Analysis functionality: The script includes functions for analyzing the combined dataset, such as calculating
   correlations, aggregating data, and visualizing results.
9. main function: The main function of the script is responsible for parsing command-line arguments, handling
   conflicting arguments, setting default values, and executing the appropriate actions based on the provided arguments.

Maintainability: 

In the event that the website or REST API changes in the future, the Python script will need to be updated accordingly
to adapt to the new structure or endpoint. The following are some general guidelines on how to modify the code:

1. Update the URLs and endpoints: If the URLs of the websites being scraped or the endpoints of the REST APIs have
   changed, you will need to update the corresponding variables in the script. For example, if the URL of Weather
   Underground's historical data changes, you should modify the `URL_WUG_MONTHLY` variable accordingly.
2. Update the parsing logic: If the structure of the website's HTML or the JSON data returned by the API changes, you
   will need to update the parsing logic. For instance, if the Weather Underground website changes the way it organizes
   its tables or adds new columns, you may need to modify the `WUGWeatherDataset.fetch_data` method and adjust the
   BeautifulSoup selectors to extract the correct data.
3. Update the data processing and transformation: If the data format or schema changes, you might need to update the
   data processing and transformation logic. For example, if the incident dataset starts returning additional fields,
   you may need to update the `IncidentDataset.fetch_data` method and adjust the DataFrame processing steps.
4. Handle new or deprecated features: If the website or REST API introduces new features or deprecates existing ones,
   you may need to update the script to incorporate or remove the related functionality. For instance, if the incident
   dataset API starts requiring authentication, you might need to add code to handle authentication and update the API
   request logic accordingly.
5. Update error handling: When the website or REST API changes, you may need to revise the error handling mechanisms to
   account for new error codes or messages. This could involve updating the conditions in exception handling blocks or
   adding new ones to handle specific errors.
6. Keep documentation up-to-date: It is crucial to keep the documentation up-to-date with the changes in the code. This
   helps users understand the new functionality, and it also serves as a reference for future developers who might need
   to maintain or modify the code.

When a website or REST API changes, it is important to thoroughly review the code and make necessary adjustments to
ensure the script continues to function correctly. By following these guidelines, you can effectively adapt the code to
handle changes and maintain the script's functionality.












