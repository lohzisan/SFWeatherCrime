# Los Angeles Traffic Data and Apartment Price Analysis Tool

## 1 Introduction

This Python script is designed to fetch, analyze, and display traffic data and apartment prices in Los Angeles. It
focuses on scraping data from the Weather Underground and San Francisco Police Department (SFPD) incident datasets. The
script offers various functionalities, such as loading and saving datasets in different formats, displaying datasets in
a tabular format, and fetching data using APIs or web scraping. The tool can be operated in three modes: static, scrape,
and default.

## 2 Installation

### 2.1. Prerequisites

- Python 3.9+
- [pipenv](https://pipenv.pypa.io/en/latest/)
- Chrome browser
- Terminal (with ANSI color support)

### 2.2 How to set up

1. Unzip the project folder
2. Open a terminal and navigate to the project folder
3. Run `pipenv install` to install dependencies (check `2.3` for more details)
4. Run `pipenv shell` to activate the virtual environment
5. Run `python main.py` to start the program
   > You may use arguments specified in next section.

### 2.3 Required third-party packages can be found in Pipfile

```toml
[packages]
pandas = "*"
requests = "*"
beautifulsoup4 = "*"
tabulate = "*"
rich = "*"
pyyaml = "*"
selenium = "*"
webdriver-manager = "*"
```

## 3 How to use

Use `python main.py --help` to see the available arguments.

There are three modes in which the script can be executed:

1. Static mode: Load a local dataset file using `--static` argument.
   Example: `python main.py --static ./data/noaa.csv`

2. Scrape mode: Fetch sample data from online sources using the `--scrape` argument.\
   Example: `python main.py --scrape`

3. Default mode: Fetch the full dataset from online sources without providing any specific mode argument.\
   Example: `python main.py`

Additional arguments are available:
- `--start-date`: Specify the start date for scraping (default: `2019-01-01`)
- `--end-date`: Specify the end date for scraping (default: `2021-01-31`)
- `--display-limit`: Limit the number of rows to display (default: `10`)
- `--random`: Display random rows from the dataset (default: `False`)
- `--name`: Specify the dataset name (choices: `noaa`, `wug`, `incident`, `auto`; default: `auto`)

```bash
python main.py --static ./data/noaa.csv
```

![Screenshot Static Mode](./screenshots/static-mode.png)

Scrape mode:

```bash
python main.py --scrape
```

![Screenshot Scrape Mode](./screenshots/scrape-mode.png)

Default mode:

```bash
python main.py
```

![Screenshot Default Mode](./screenshots/default-mode.png)

## 4 Datasets

You may find dataset inside the `data` folder.
Incidents dataset is too larger therefore it is not included in the repository.
[Google Drive : Incident List](https://drive.google.com/file/d/1zYH68NvS6a58BaLgZ23DXjTyUpqDGfUp/view?usp=sharing)

## 5 Architecture

This Python script has several components and is used for scraping and fetching specific datasets.
It is designed to fetch data from the Weather Underground and San Francisco Police Department (SFPD) incident datasets.
The script provides several functionalities, such as loading and saving datasets from/to different file formats,
displaying datasets in a table format, and fetching data from APIs or by scraping websites.

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
7. main function: The main function of the script is responsible for parsing command-line arguments, handling
   conflicting arguments, setting default values, and executing the appropriate actions based on the provided arguments.

## 6 Maintenance

x