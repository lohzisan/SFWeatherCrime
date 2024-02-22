#!/bin/python3
import argparse
import json
import os
import time

import bs4
import pandas as pd
import requests
import yaml
from rich.console import Console
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tabulate import tabulate
from webdriver_manager.chrome import ChromeDriverManager

console = Console()

# Configurations
# --------------------------------------------------
URL_WUG_MONTHLY = "https://www.wunderground.com/history/monthly/us/ca/san-francisco/KSFO/date/"
URL_INCIDENT_BASE = "https://data.sfgov.org/resource/wg3w-h783.json"
CHROME_BINARY_PATH = None  # "/usr/bin/google-chrome-stable"


# Utility functions
# --------------------------------------------------

def snake_to_normal_case(text):
    """
    Convert snake case to normal case
    :param text: str    :return: str
    """
    text = text.replace("_", " ")
    text = text.title()
    return text


def get_file_type(file):
    """
    Get file type from file extension
    :param file: str    :return: str
    """
    file_extension = os.path.splitext(file)[1]
    if file_extension == '.csv':
        return 'csv'
    elif file_extension == '.json':
        return 'json'
    elif file_extension == '.yaml':
        return 'yaml'
    else:
        console.print("Unsupported file format", style="bold red")


def chrome_get_htmls(urls, chrome_binary_path=CHROME_BINARY_PATH):
    """
    Get htmls from urls using chrome. This is used to get data from websites that use javascript to render the data.
    :param chrome_binary_path: is the path to the chrome binary. If not set, it will use the default chrome binary.
    :param urls: list    :return: list
    """
    # create a new chrome session
    console.print("Starting Chrome browser")
    service = Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    if CHROME_BINARY_PATH and os.path.exists(chrome_binary_path):
        options.binary_location = CHROME_BINARY_PATH
    options.add_argument("--headless")
    browser = webdriver.Chrome(service=service, options=options)

    def get_html(url):
        console.print(f"Getting data from {url}")
        browser.get(url)
        # wait for the page to load
        xpath_expression = '//*[@aria-labelledby="History days"]'
        element = WebDriverWait(browser, 60).until(
            EC.presence_of_element_located((By.XPATH, xpath_expression))
        )
        if not element:
            console.print("Failed to load page")
            return None
        return browser.page_source

    htmls = [get_html(url) for url in urls]

    browser.quit()
    return htmls


class Dataset:
    def __init__(self):
        """
        Base class for datasets
        """
        self.df = None
        self.display_cols = []
        self.table_style = 'simple'

    @classmethod
    def from_dataset(cls, dataset):
        """
        Copy data from another dataset
        :param dataset: Dataset
        :return:
        """
        new_dataset = cls()
        new_dataset.df = dataset.df.copy()
        return new_dataset

    def load_data(self, path, file_type='auto'):
        """
        Load data from file
        :param path: str - path to file
        :param file_type: str - file type. If 'auto', it will be inferred from the file extension
        :return:
        """
        if file_type == 'auto':
            file_type = get_file_type(path)
        if file_type == 'csv':
            self.df = pd.read_csv(path)
        elif file_type == 'json':
            with open(path, 'r') as f:
                data = json.load(f)
            self.df = pd.DataFrame(data)
        elif file_type == 'yaml':
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            self.df = pd.DataFrame(data)
        else:
            console.print("Unsupported file format", style="bold red")

    def save_data(self, path, file_type='csv'):
        """
        Save data to file
        :param path: str - path to file
        :param file_type: str - file type. default is csv
        :return:
        """
        if self.df is None:
            console.print("No data to save", style="bold red")
            return
        if file_type == 'csv':
            self.df.to_csv(path, index=False)
        elif file_type == 'json':
            with open(path, 'w') as f:
                json.dump(self.df.to_dict(orient='records'), f)
        elif file_type == 'yaml':
            with open(path, 'w') as f:
                yaml.dump(self.df.to_dict(orient='records'), f)
        else:
            console.print("Unsupported file format", style="bold red")

    def display(self, display_limit=10, offset=0, cols=None, random=False):
        """
        Display data in a table
        :param display_limit: number of rows to display
        :param offset: offset from the start
        :param cols: list of columns to display
        :param random: if True, display random rows
        :return:
        """
        if self.df is None:
            console.print("No data to display", style="bold red")
            return
        if random:
            sample = self.df.sample(n=display_limit)
        else:
            sample = self.df.iloc[offset:display_limit + offset]
        cols = cols or self.display_cols

        # check if all display columns exist in the dataframe
        df_cols = set(self.df.columns)
        display_cols = set(self.display_cols)
        if not display_cols.issubset(df_cols):
            console.print("Some display columns do not exist in the dataframe", style="bold red")
            cols = list(df_cols.intersection(display_cols))

        sample = sample[cols]
        sample.columns = [snake_to_normal_case(col) for col in sample.columns]

        # format datetime or date columns to display in a more readable format
        for col in sample.columns:
            if 'datetime' in sample[col].dtype.name:
                sample[col] = sample[col].dt.strftime('%Y-%m-%d %H:%M:%S')

        console.print(tabulate(sample, headers='keys', tablefmt=self.table_style, showindex=False, maxcolwidths=30))
        console.print(f"Total rows: {len(self.df)} | Total columns: {len(self.df.columns)}", style="bold")

    def is_display_cols_exists(self):
        """
        Check if all display columns exist in the dataframe
        :return:
        """
        df_cols = set(self.df.columns)
        display_cols = set(self.display_cols)
        return display_cols.issubset(df_cols)


class NOAAWeatherDataset(Dataset):

    def __init__(self):
        """
        NOAA weather dataset
        """
        super().__init__()
        self.display_cols = ['STATION', 'NAME', 'DATE', 'PRCP', 'RAIN', 'TAVG', 'TMAX', 'TMIN']


class WUGWeatherDataset(Dataset):
    def __init__(self):
        """
        Weather Underground weather dataset
        """
        super().__init__()
        self.raw_header_cols = ['time', 'temp', 'dew_point', 'humidity', 'wind_speed',
                                'pressure', 'precipitation']
        self.display_cols = ['date', 'temp_avg', 'dew_point_avg', 'humidity_avg', 'wind_speed_avg',
                             'pressure_avg']

    def fetch_data(self, start_date, end_date, limit=None):
        """
        Scrape data from weather underground website
        :param start_date: str - start date in YYYY-MM-DD format
        :param end_date: str - end date in YYYY-MM-DD format
        :param limit: int - number of rows to fetch, if None, fetch all rows
        :return:
        """

        def to_df(html):
            if html is None:
                return None
            soup = bs4.BeautifulSoup(html, "html.parser")
            daily_obs_table = soup.find(
                lambda tag: tag.has_attr('aria-labelledby') and tag['aria-labelledby'] == 'History days')
            sub_tables = daily_obs_table.findAll(lambda tag: tag.name == 'table')
            dfs = []
            for i, table in enumerate(sub_tables):
                h1 = self.raw_header_cols[i]
                rows = table.findAll(lambda tag: tag.name == 'tr')
                if i == 0:
                    data = {'': []}
                else:
                    data = {r: [] for r in rows[0].text.split()}
                for row in rows[1:]:
                    for i, cell in enumerate(row.findAll(lambda tag: tag.name == 'td')):
                        data[list(data.keys())[i]].append(cell.text)
                df = pd.DataFrame(data)
                df.columns = pd.MultiIndex.from_product([[h1], df.columns])
                dfs.append(df)

            return pd.concat(dfs, axis=1)

        console.print(f"Scraping data from {start_date} to {end_date}")
        months = pd.date_range(start_date, end_date, freq='MS').strftime("%Y-%m").tolist()
        if limit:
            months = months[:limit]
        urls = [URL_WUG_MONTHLY + month for month in months]
        htmls = chrome_get_htmls(urls)
        dfs = [to_df(html) for html in htmls]

        for i, df in enumerate(dfs):
            if df is None:
                continue
            df.insert(0, 'month', months[i])

        dfs = list(filter(lambda x: x is not None, dfs))
        df = pd.concat(dfs, axis=0)
        df.columns = ['_'.join(col).strip(' _') for col in df.columns.values]
        df.columns = [col.lower() for col in df.columns]

        df.insert(0, 'date', df['month'].astype(str) + '-' + df['time'].astype(str))
        df.drop(['month', 'time'], axis=1, inplace=True)
        df['date'] = df['date'].apply(lambda x: pd.to_datetime(str(x).replace(' ', ''), format='%Y-%m-%d'))
        df['date'] = df['date'].dt.date

        console.print(f"Scraped {len(df)} rows of data")
        self.df = df


class IncidentDataset(Dataset):
    def __init__(self):
        """
        San Francisco Police Department incident dataset
        """
        super().__init__()
        self.display_cols = ['incident_id', 'incident_datetime', 'incident_category', 'resolution', 'police_district']

    def fetch_data(self, start_date=None, end_date=None, limit=None):
        """
        Fetch data from SFPD API
        :param start_date: str - start date in YYYY-MM-DD format
        :param end_date: str - end date in YYYY-MM-DD format
        :param limit: int - number of rows to fetch, if None, fetch all rows
        :return:
        """

        def fetch_data_from_api(_url, _params):
            response = requests.get(_url, params=_params)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data)
                # df['incident_datetime'] = df['incident_datetime'].apply(lambda x: pd.to_datetime(x))
                return df
            else:
                console.print(f"Error fetching data: {response.status_code}", style="bold red")
                return None

        console.print(f"Fetching data from {start_date} to {end_date}")
        base_url = URL_INCIDENT_BASE
        params = {}
        if start_date is not None and end_date is not None:
            params['$where'] = f"incident_datetime >= '{start_date}T00:00:00.000' AND " \
                               f"incident_datetime <= '{end_date}T23:59:59.999'"
        elif start_date is not None:
            params['$where'] = f"incident_datetime >= '{start_date}T00:00:00.000'"
        elif end_date is not None:
            params['$where'] = f"incident_datetime <= '{end_date}T23:59:59.999'"

        dfs = []
        step_size = 5000
        offset = 0
        local_limit = min(step_size, limit or step_size)
        while True:
            params['$offset'] = offset
            params['$limit'] = local_limit
            console.print(f"Fetching {offset} to {offset + local_limit} rows of data")
            df = fetch_data_from_api(base_url, params)
            if df is None or df.shape[0] == 0:
                break
            dfs.append(df)
            if limit is not None:
                limit -= df.shape[0]
                if limit <= 0:
                    break
            offset += local_limit

        self.df = pd.concat(dfs, axis=0)
        console.print(f"Successfully fetched {len(self.df)} rows of data")


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='San Francisco Crime Reports and Weather Data Tool')
    parser.add_argument('--static', type=str, help='Path to static data file')
    parser.add_argument('--scrape', action='store_true', help='Scrape data from website')
    parser.add_argument('--start-date', type=str, help='Start date for scraping', default='2019-01-01')
    parser.add_argument('--end-date', type=str, help='End date for scraping', default='2021-01-31')
    parser.add_argument('--display-limit', type=int, help='Limit for scraping', default=10)
    parser.add_argument('--random', action='store_true', help='Display random rows', default=False)
    parser.add_argument('--name', type=str, help="Dataset name, 'noaa', 'wug', 'incident', 'auto'. Default is 'auto'",
                        default='auto',
                        choices=['noaa', 'wug', 'incident', 'auto'])

    args = parser.parse_args()

    # check for conflicting arguments
    if args.static is not None and args.scrape:
        console.print("Cannot specify both static and scrape", style="bold red")
        return

    start_time = time.time()

    # actions
    if args.static is not None:
        # check if file exists
        if not os.path.exists(args.static):
            console.print(f"File {args.static} does not exist", style="bold red")
            return
        if args.name == 'auto':
            dataset = NOAAWeatherDataset()
            dataset.load_data(args.static)

            if not dataset.is_display_cols_exists():
                dataset = WUGWeatherDataset.from_dataset(dataset)

            if not dataset.is_display_cols_exists():
                dataset = IncidentDataset.from_dataset(dataset)

            if not dataset.is_display_cols_exists():
                console.print(f"Cannot load data from {args.static}", style="bold red")
                return
        elif args.name == 'noaa':
            dataset = NOAAWeatherDataset()
            dataset.load_data(args.static)
        elif args.name == 'wug':
            dataset = WUGWeatherDataset()
            dataset.load_data(args.static)
        elif args.name == 'incident':
            dataset = IncidentDataset()
            dataset.load_data(args.static)
        else:
            console.print(f"Unknown dataset name {args.name}", style="bold red")
            return

        dataset.display(random=args.random, display_limit=args.display_limit)

    elif args.scrape:
        if args.name not in ['wug', 'incident', 'auto']:
            console.print(f"Unsupported dataset name {args.name}", style="bold red")
            return
        if args.name == 'wug' or args.name == 'auto':
            console.print("\nScraping weather data", style="bold blue")
            dataset = WUGWeatherDataset()
            dataset.fetch_data(args.start_date, args.end_date, limit=5)  # limit=5*30
            dataset.display(random=args.random, display_limit=args.display_limit)
        if args.name == 'incident' or args.name == 'auto':
            console.print("\nFetching incident data", style="bold blue")
            dataset = IncidentDataset()
            dataset.fetch_data(args.start_date, args.end_date, limit=150)
            dataset.display(random=args.random, display_limit=args.display_limit)
    else:
        if args.name not in ['wug', 'incident', 'auto']:
            console.print(f"Unsupported dataset name {args.name}", style="bold red")
            return
        if args.name == 'wug' or args.name == 'auto':
            console.print("\nScraping weather data", style="bold blue")
            wug_dataset = WUGWeatherDataset()
            wug_dataset.fetch_data(args.start_date, args.end_date)
            console.print("\nSaving weather data", style="bold blue")
            wug_dataset.save_data('data/weather.csv', file_type='csv')
            wug_dataset.display(random=args.random, display_limit=args.display_limit)
        if args.name == 'incident' or args.name == 'auto':
            console.print("\nFetching incident data", style="bold blue")
            inc_dataset = IncidentDataset()
            inc_dataset.fetch_data(args.start_date, args.end_date)
            console.print("\nSaving incident data", style="bold blue")
            inc_dataset.save_data('data/incidents.csv', file_type='csv')
            inc_dataset.display(random=args.random, display_limit=args.display_limit)

    end_time = time.time()
    console.print()
    console.print(f"Total time taken: {end_time - start_time:.4f} seconds", style="bold yellow")


if __name__ == "__main__":
    main()
