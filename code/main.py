#!/bin/python3
import argparse
import json
import os
import time

import bs4
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import yaml
from rich.console import Console
from scipy.stats import ttest_ind, pearsonr, kendalltau, spearmanr
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
CHROME_BINARY_PATH = None  # "/usr/bin/brave"
NOAA_DATA_PATH = 'data/noaa.csv'
WUG_DATA_PATH = 'data/weather.csv'
INCIDENT_DATA_PATH = 'data/incidents.csv'
COMBINED_DATA_PATH = 'data/combined.csv'


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
    if chrome_binary_path and os.path.exists(chrome_binary_path):
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


class Timer:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.elapsed = self.end - self.start
        console.print()
        console.print(f"{self.message} took {self.elapsed:.2f} seconds", style="bold yellow")


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

    def cast_to_dataset(self, *Classes):
        """
        Cast dataset to another dataset class
        :param Classes: list of Dataset classes
        :return:
        """
        for Cls in Classes:
            empty_dataset = Cls()
            empty_dataset.df = self.df.copy()
            if empty_dataset.is_display_cols_exists():
                return empty_dataset
        return None

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


class CombinedDataset(Dataset):
    def __init__(self):
        """
        Combined dataset using NOAA, WUG, and Incident datasets
        """
        super().__init__()
        self.display_cols = ['Date', 'Rain', 'Average Temp (NOAA)', 'Average Temp (Wunderland)',
                             'Number of Crimes', 'Average Temp (Combined)']

    @classmethod
    def from_datasets(cls, noaa_dataset, wug_dataset, incident_dataset):
        dataset = cls()
        dataset.combine_datasets(noaa_dataset, wug_dataset, incident_dataset)
        return dataset

    def combine_datasets(self, noaa_ds, wug_ds, incident_ds):
        # Copy the DataFrames from each dataset
        df_noaa = noaa_ds.df.copy()
        df_weather = wug_ds.df.copy()
        df_incidents = incident_ds.df.copy()

        # Prepare dataframes
        df_incidents['DATE'] = pd.to_datetime(df_incidents['incident_date']).dt.strftime('%m/%d/%Y')
        grouped = df_incidents.groupby('DATE').size()
        df_incidents = pd.DataFrame({
            'Date': grouped.index,
            'Number of Crimes': grouped.values
        })
        df_noaa['DATE'] = pd.to_datetime(df_noaa['DATE'], format='%m/%d/%y').dt.strftime('%m/%d/%Y')
        df_noaa = pd.DataFrame({
            'Date': df_noaa['DATE'],
            'Rain': df_noaa['RAIN'],
            'Average Temp (NOAA)': df_noaa['TAVG']
        })
        df_weather['date'] = pd.to_datetime(df_weather['date']).dt.strftime('%m/%d/%Y')
        df_weather = pd.DataFrame({
            'Date': df_weather['date'],
            'Average Temp (Wunderland)': df_weather['temp_avg'],
        })

        # Merge the DataFrames
        combined_df = df_incidents.merge(df_noaa, on='Date', how='left').merge(df_weather, on='Date', how='left')
        combined_df['Average Temp (Combined)'] = \
            combined_df[['Average Temp (NOAA)', 'Average Temp (Wunderland)']].mean(axis=1)

        # Convert the date column to datetime
        combined_df['Date'] = pd.to_datetime(combined_df['Date'], format='%m/%d/%Y')
        self.df = combined_df

    def scatter_plot_by_year(self, output_dir='analysis'):
        console.print("\nAnalysis: Average Temp, Number of Crimes by Year", style="bold blue")

        # prepare data
        df = self.df.copy()
        df['Year'] = df['Date'].dt.year

        # plot
        sns.set_style('whitegrid')
        sns.lmplot(x='Average Temp (Combined)', y='Number of Crimes', data=df, hue='Year', palette='bright')
        plt.xlabel('Average Temperature')
        plt.ylabel('Number of Crimes')

        # save plot
        plt.savefig(os.path.join(output_dir, 'scatter_plot_by_year.png'))
        plt.close()
        console.print(f"Saved scatter plot by year to {output_dir}/scatter_plot_by_year.png")

    def bar_chart_rainy_non_rainy_days(self, output_dir='analysis'):
        console.print("\nAnalysis: Number of Crimes on Rainy and Non-Rainy Days", style="bold blue")

        # prepare data
        df = self.df.copy()
        rainy_crimes = df[df['Rain'] == 'YES']['Number of Crimes'].sum()
        non_rainy_crimes = df[df['Rain'] == 'NO']['Number of Crimes'].sum()
        crimes_df = pd.DataFrame({'Rainy Days': rainy_crimes, 'Non-Rainy Days': non_rainy_crimes}, index=[0])

        # plot
        ax = crimes_df.plot(kind='bar', color=['blue', 'green'])
        ax.set_xlabel('Rain')
        ax.set_ylabel('Number of Crimes')
        ax.set_title('Number of Crimes on Rainy and Non-Rainy Days')
        rects = ax.patches
        labels = [rainy_crimes, non_rainy_crimes]
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label, ha='center', va='bottom')

        # save plot
        plt.savefig(os.path.join(output_dir, 'bar_chart_rainy_non_rainy_days.png'))
        plt.close()
        console.print(f"Saved bar chart of rainy and non-rainy days to {output_dir}/bar_chart_rainy_non_rainy_days.png")

    def t_test_pearsons_correlation(self, output_dir='analysis', threshold=65):
        console.print("\nAnalysis: T-Test and Pearson's Correlation of "
                      "Average Temperature (Combined) vs. Number of Crimes", style="bold blue")

        # prepare data
        df = self.df.copy()

        # temperature threshold
        high_temp = df.loc[df['Average Temp (Combined)'] > threshold, 'Number of Crimes']
        low_temp = df.loc[df['Average Temp (Combined)'] <= threshold, 'Number of Crimes']
        temp_t_stat, temp_p_val = ttest_ind(high_temp, low_temp, equal_var=True)
        console.print(f"T-Test of Low vs. High Temperature (threshold={threshold})")
        console.print(f"T-statistic={temp_t_stat:.5f}, p-value={temp_p_val:.5f}")

        # print t-test and pearson's correlation
        t_stat, p_val = ttest_ind(df["Average Temp (Combined)"], df["Number of Crimes"])
        console.print(f"T-Test of Average Temperature vs. Number of Crimes")
        console.print(f"T-statistic={t_stat:.5f}, p-value={p_val:.5f}")
        corr, p = pearsonr(df["Average Temp (Combined)"], df["Number of Crimes"])
        console.print(f"Pearson's Correlation of Average Temperature vs. Number of Crimes")
        console.print(f"Pearson's Correlation={corr:.5f}, p-value={p:.5f}")

        # save results
        with open(os.path.join(output_dir, 't_test_pearsons_correlation.txt'), 'w') as f:
            f.write(f"T-Test of Low vs. High Temperature (threshold={threshold})\n")
            f.write(f"T-statistic={temp_t_stat}, p-value={temp_p_val}\n")
            f.write(f"T-Test of Average Temperature vs. Number of Crimes\n")
            f.write(f"T-statistic={t_stat}, p-value={p_val}\n")
            f.write(f"Pearson's Correlation of Average Temperature vs. Number of Crimes\n")
            f.write(f"Pearson's Correlation={corr}, p-value={p}\n")
        console.print(f"Saved t-test and pearson's correlation to {output_dir}/t_test_pearsons_correlation.txt")

    def t_test_rain_pearsons_correlation(self, output_dir='analysis'):
        console.print("\nAnalysis: T-Test and Pearson's Correlation of Rain vs. Number of Crimes", style="bold blue")

        # prepare data
        df = self.df.copy()

        # print t-test and pearson's correlation
        rain_yes = df[df['Rain'] == 'YES']['Number of Crimes']
        rain_no = df[df['Rain'] == 'NO']['Number of Crimes']
        t_stat, p_val = ttest_ind(rain_yes, rain_no, equal_var=False)
        console.print(f"T-Test of Rain vs. Number of Crimes")
        console.print(f'T-statistic={t_stat:.5f}, p-value={p_val:.5f}')

        corr, p_val = pearsonr(df['Rain'].map({'YES': 1, 'NO': 0}), df['Number of Crimes'])
        console.print(f"Pearson's Correlation of Rain vs. Number of Crimes")
        console.print(f"Pearson's Correlation={corr:.5f}, p-value={p_val:.5f}")

        # save results
        with open(os.path.join(output_dir, 't_test_rain_pearsons_correlation.txt'), 'w') as f:
            f.write(f"T-Test of Rain vs. Number of Crimes\n")
            f.write(f"T-statistic={t_stat}, p-value={p_val}\n")
            f.write(f"Pearson's Correlation of Rain vs. Number of Crimes\n")
            f.write(f"Pearson's Correlation={corr}, p-value={p_val}\n")
        console.print(f"Saved t-test and pearson's correlation to analysis/t_test_rain_pearsons_correlation.txt")

    def kendalls_correlation(self, output_dir='analysis'):
        console.print("\nAnalysis: Kendall's Correlation", style="bold blue")

        # prepare data
        df = self.df.copy()

        # print kendall's correlation
        corr1, p_val1 = kendalltau(df['Number of Crimes'], df['Average Temp (Combined)'])
        console.print("Kendall's correlation between Number of Crimes and Average Temperature (Combined): ")
        console.print(f'Correlation={corr1:.5f}, p-value={p_val1:.5f}')

        corr2, p_val2 = kendalltau(df['Rain'], df['Number of Crimes'])
        console.print("Kendall's correlation between RAIN and Number of Crimes:")
        console.print(f'Correlation={corr2:.5f}, p-value={p_val2:.5f}')

        # save results
        with open(os.path.join(output_dir, 'kendalls_correlation.txt'), 'w') as f:
            f.write(f"Kendall's correlation between Number of Crimes and Average Temperature (Combined):\n")
            f.write(f'Correlation={corr1}, p-value={p_val1}\n')
            f.write(f"Kendall's correlation between RAIN and Number of Crimes:\n")
            f.write(f'Correlation={corr2}, p-value={p_val2}\n')
        console.print(f"Saved Kendall's correlation to analysis/kendalls_correlation.txt")

    def spearmans_correlation(self, output_dir='analysis'):
        console.print("\nAnalysis: Spearman's Correlation", style="bold blue")

        # prepare data
        df = self.df.copy()

        # print spearman's correlation
        corr1, pval1 = spearmanr(df['Number of Crimes'], df['Average Temp (Combined)'])
        console.print(f"Spearman's correlation between Number of Crimes and Average Temperature (Combined):")
        console.print(f"correlation={corr1:.5f}, p-value={pval1:.5f}")

        corr2, pval2 = spearmanr(df['Rain'], df['Number of Crimes'])
        console.print(f"Spearman's correlation between RAIN and Number of Crimes:")
        console.print(f"correlation={corr2:.5f}, p-value={pval2:.5f}")

        # save results
        with open(os.path.join(output_dir, 'spearmans_correlation.txt'), 'w') as f:
            f.write(f"Spearman's correlation between Number of Crimes and Average Temperature (Combined): "
                    f"correlation={corr1}, p-value={pval1}\n")
            f.write(f"Spearman's correlation between RAIN and Number of Crimes: "
                    f"correlation={corr2}, p-value={pval2}\n")
        console.print(f"Saved Spearman's correlation to analysis/spearmans_correlation.txt")

    def run_analysis(self):
        self.scatter_plot_by_year()
        self.bar_chart_rainy_non_rainy_days()
        self.t_test_pearsons_correlation()
        self.t_test_rain_pearsons_correlation()
        self.kendalls_correlation()
        self.spearmans_correlation()


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='San Francisco Crime Rate and Weather Analysis Tool')
    parser.add_argument('--static', type=str, help='Path to static data file')
    parser.add_argument('--scrape', action='store_true', help='Scrape data from website')
    parser.add_argument('--analyze', action='store_true', help='Analyze data')
    parser.add_argument('--start-date', type=str, help='Start date for scraping', default='2019-01-01')
    parser.add_argument('--end-date', type=str, help='End date for scraping', default='2021-01-31')
    parser.add_argument('--display-limit', type=int, help='Limit for scraping', default=10)
    parser.add_argument('--random', action='store_true', help='Display random rows', default=False)
    parser.add_argument('--name', type=str, help="Dataset name, 'noaa', 'wug', 'incident', 'auto'. Default is 'auto'",
                        default='auto',
                        choices=['noaa', 'wug', 'incident', 'combined', 'auto'])

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

        with Timer("Loading data from static file"):
            if args.name == 'auto':
                dataset = Dataset()
                dataset.load_data(args.static)
                dataset = dataset.cast_to_dataset(NOAAWeatherDataset, WUGWeatherDataset,
                                                  IncidentDataset, CombinedDataset)
                if dataset is None:
                    console.print(f"Unknown dataset", style="bold red")
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
            elif args.name == 'combined':
                dataset = CombinedDataset()
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
            with Timer("Scraping weather data"):
                console.print("\nScraping weather data", style="bold blue")
                combined_dataset = WUGWeatherDataset()
                combined_dataset.fetch_data(args.start_date, args.end_date, limit=5)  # limit=5*30
                combined_dataset.display(random=args.random, display_limit=args.display_limit)
        if args.name == 'incident' or args.name == 'auto':
            with Timer("Scraping incident data"):
                console.print("\nFetching incident data", style="bold blue")
                combined_dataset = IncidentDataset()
                combined_dataset.fetch_data(args.start_date, args.end_date, limit=150)
                combined_dataset.display(random=args.random, display_limit=args.display_limit)

    elif args.analyze:
        # load all 3 datasets
        with Timer("Loading data from static file"):
            console.print("Loading NOAA data", style="bold blue")
            noaa_dataset = NOAAWeatherDataset()
            noaa_dataset.load_data(NOAA_DATA_PATH)
            console.print("Loading WUG data", style="bold blue")
            wug_dataset = WUGWeatherDataset()
            wug_dataset.load_data(WUG_DATA_PATH)
            console.print("Loading incident data", style="bold blue")
            incident_dataset = IncidentDataset()
            incident_dataset.load_data(INCIDENT_DATA_PATH)

        # merge datasets
        with Timer("Merging datasets"):
            combined_dataset = CombinedDataset.from_datasets(noaa_dataset=noaa_dataset, wug_dataset=wug_dataset,
                                                             incident_dataset=incident_dataset)
            combined_dataset.display(random=args.random, display_limit=args.display_limit)
            combined_dataset.save_data(COMBINED_DATA_PATH, file_type='csv')

        # run analysis
        with Timer("Running analysis"):
            combined_dataset.run_analysis()

    else:
        if args.name not in ['wug', 'incident', 'auto']:
            console.print(f"Unsupported dataset name {args.name}", style="bold red")
            return
        if args.name == 'wug' or args.name == 'auto':
            with Timer("Scraping weather data"):
                console.print("\nScraping weather data", style="bold blue")
                wug_dataset = WUGWeatherDataset()
                wug_dataset.fetch_data(args.start_date, args.end_date)
                console.print("\nSaving weather data", style="bold blue")
                wug_dataset.save_data(WUG_DATA_PATH, file_type='csv')
                wug_dataset.display(random=args.random, display_limit=args.display_limit)
        if args.name == 'incident' or args.name == 'auto':
            with Timer("Scraping incident data"):
                console.print("\nFetching incident data", style="bold blue")
                inc_dataset = IncidentDataset()
                inc_dataset.fetch_data(args.start_date, args.end_date)
                console.print("\nSaving incident data", style="bold blue")
                inc_dataset.save_data(INCIDENT_DATA_PATH, file_type='csv')
                inc_dataset.display(random=args.random, display_limit=args.display_limit)

        if args.name == 'auto':
            # ask user if they want to merge datasets
            console.print("\nDo you want to merge datasets? (y/n)", style="bold blue", end=" ")
            merge = input()
            if merge != 'y':
                return

            # load all 3 datasets
            with Timer("Loading data from static file"):
                console.print("Loading NOAA data", style="bold blue")
                noaa_dataset = NOAAWeatherDataset()
                noaa_dataset.load_data(NOAA_DATA_PATH)
                console.print("Loading WUG data", style="bold blue")
                wug_dataset = WUGWeatherDataset()
                wug_dataset.load_data(WUG_DATA_PATH)
                console.print("Loading incident data", style="bold blue")
                incident_dataset = IncidentDataset()
                incident_dataset.load_data(INCIDENT_DATA_PATH)

            # merge datasets
            with Timer("Merging datasets"):
                console.print("\nMerging datasets", style="bold blue")
                combined_dataset = CombinedDataset.from_datasets(noaa_dataset=noaa_dataset, wug_dataset=wug_dataset,
                                                                 incident_dataset=incident_dataset)
                combined_dataset.display(random=args.random, display_limit=args.display_limit)
                combined_dataset.save_data(COMBINED_DATA_PATH, file_type='csv')
                console.print("\nSaving merged dataset", style="bold blue")
                console.print("Run --analyze to run analysis on merged dataset", style="bold blue")

            # ask user if they want to run analysis
            console.print("\nDo you want to run analysis on merged dataset? (y/n)", style="bold blue", end=" ")
            analyze = input()
            if analyze != 'y':
                return

            # run analysis
            with Timer("Running analysis"):
                combined_dataset.run_analysis()

    end_time = time.time()
    console.print(f"Total time taken: {end_time - start_time:.4f} seconds", style="bold yellow")


if __name__ == "__main__":
    main()
