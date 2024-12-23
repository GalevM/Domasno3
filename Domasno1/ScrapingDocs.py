import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os
from multiprocessing import Manager, Pool

base_url = "https://www.mse.mk/mk/stats/symbolhistory/REPL"

today = datetime.now()
data_rows = []
flag = False

if os.path.exists("dokss.csv"):
    flagExists = True
else:
    flagExists = False


def fetch_data(code, data):
    url = f"https://www.mse.mk/mk/stats/symbolhistory/{code}/"
    response = requests.post(url, data=data)
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")
    if not table:
        return []

    rows = table.find_all("tr")
    code_data = []

    for row in rows[1:]:
        cols = row.find_all("td")
        cols = [col.text.strip() for col in cols]
        cols.append(code)
        code_data.append(cols)

    return code_data


def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)


def get_codes(soup):
    codesGot = soup.find("select")
    codesGot = codesGot.text
    codesGot = codesGot.split("\n")
    filtered_codes = []
    for code in codesGot:
        if has_numbers(code):
            continue
        else:
            filtered_codes.append(code)

    return filtered_codes[:-1]


response = requests.get(base_url)
soup = BeautifulSoup(response.text, "html.parser")
codes = get_codes(soup)


def get_existing_dates_from_csv():
    if os.path.exists("dokss.csv"):
        df = pd.read_csv("dokss.csv", parse_dates=["Datum"], dayfirst=True)
        return set(df["Datum"].dt.date)
    else:
        return set


def scrape(data, manager_data_rows):
    with ThreadPoolExecutor() as executor:
        future_to_code = {executor.submit(fetch_data, code, data): code for code in codes}

        for future in as_completed(future_to_code):
            code_data = future.result()

            manager_data_rows.extend(code_data)


def process_year(year, manager_data_rows, existing_dates):
    if flagExists:
        if year >= 1:
            return

        dateLast = max(existing_dates)
        todayTmp = pd.to_datetime(today, format="%d.%m.%Y", errors='coerce')
        dateLast = pd.to_datetime(dateLast, format="%d.%m.%Y", errors='coerce')
        dateLast = dateLast.strftime("%d.%m.%Y")
        todayTmp = todayTmp.strftime("%d.%m.%Y")
        if dateLast != todayTmp:
            dateLast = pd.to_datetime(dateLast, format="%d.%m.%Y", errors='coerce') + timedelta(days=1)
            missing_dates = [pd.date_range(dateLast, todayTmp)]
        else:
            missing_dates = []

        if len(missing_dates) == 0:
            print("Nothing to scrape, all the dates are scraped!")
            return

        print(f"Scraping from {dateLast} to {todayTmp}")
        data = {
            'FromDate': dateLast,
            'ToDate': todayTmp
        }

        scrape(data, manager_data_rows)

    else:
        from_date = (today - timedelta(days=365 * (year + 1))).date()
        to_date = (today - timedelta(days=365 * year)).date()

        print(f"Scraping from {from_date} to {to_date}")

        data = {
            'FromDate': from_date,
            'ToDate': to_date
        }

        scrape(data, manager_data_rows)


def main():
    start_time = time.time()

    existing_dates = get_existing_dates_from_csv()

    with Manager() as manager:
        manager_data_rows = manager.list()

        years = range(10)
        with Pool() as pool:
            pool.starmap(process_year, [(year, manager_data_rows, existing_dates) for year in years])

        data_rows = list(manager_data_rows)

    if flag:
        df = pd.read_csv("dokss.csv", parse_dates=["Datum"], dayfirst=True)
    else:
        if flagExists:
            df1 = pd.read_csv("dokss.csv", parse_dates=["Datum"], dayfirst=True)

            if len(data_rows) > 0:
                df = pd.DataFrame(data_rows)
                df.columns = ['Datum', 'Cena na posledna transakcija', 'Mak.', 'Min.', 'Prosecna cena', '%prom',
                              'Kolicina', 'Promet vo BEST vo denari', 'Vkupen promet vo denari', 'Ime na Kompanija']
                df = pd.concat([df1, df], ignore_index=True, axis=0)
            else:
                df = df1
        else:
            df = pd.DataFrame(data_rows)

    df.columns = ['Datum', 'Cena na posledna transakcija', 'Mak.', 'Min.', 'Prosecna cena', '%prom', 'Kolicina',
                  'Promet vo BEST vo denari', 'Vkupen promet vo denari', 'Ime na Kompanija']

    df["Datum"] = df["Datum"].apply(lambda x: x.strip() if isinstance(x, str) else x)
    df["Datum"] = pd.to_datetime(df["Datum"], format="%d.%m.%Y", errors='coerce')
    df.sort_values(by=["Ime na Kompanija", "Datum"], ascending=[True, False], inplace=True)

    print(df)
    df.to_csv("dokss.csv", index=False, date_format="%d-%m-%Y")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to execute: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
