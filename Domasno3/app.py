from urllib import request

import pandas as pd
from flask import Flask, render_template, request
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('Agg')
import io
import base64

app = Flask("ScrapingDocs")

dataFile = pd.read_csv("../Domasno1/dokss.csv")

issuers = [row for row in dataFile['Ime na Kompanija']]
issuers = set(issuers)
issuers = sorted(issuers)


def parsePrice(price_str):
    price_str = price_str.replace('.', '')
    price_str = price_str.replace(',', '.')
    return float(price_str)


df = dataFile[dataFile['Kolicina'] != 0]
df['%prom'] = df['%prom'].ffill()
df['Prosecna cena'] = df['Prosecna cena'].apply(parsePrice)


def generateGraph(issuer, df):
    issuer_data = df[df['Ime na Kompanija'] == issuer].copy()

    if issuer_data.empty:
        return None

    issuer_data['Datum'] = pd.to_datetime(issuer_data['Datum'], errors='coerce')
    issuer_data.loc[:, 'Prosecna cena'] = pd.to_numeric(issuer_data['Prosecna cena'], errors='coerce')

    issuer_data.dropna(subset=['Datum', 'Prosecna cena'], inplace=True)

    issuer_data.set_index('Datum', inplace=True)
    issuer_data.sort_index(inplace=True)

    issuer_data['Prosecna cena'] = issuer_data['Prosecna cena'].interpolate()

    # Generate the line chart
    plt.figure(figsize=(10, 6))
    plt.plot(
        issuer_data.index, issuer_data['Prosecna cena'],
        label=f'{issuer} Prosecna cena', color='blue', linestyle='-', linewidth=2
    )
    plt.title(f'{issuer} Просечна Цена Низ Времето')
    plt.xlabel('Datum')
    plt.ylabel('Prosecna cena')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return img_base64


@app.route('/')
def index():
    """Render the homepage."""
    return render_template('page1.html')


@app.route('/page2', methods=['GET', 'POST'])
def show_graphs():
    if request.method == 'POST':
        issuer = request.form.get('issuer')

        if issuer not in issuers:
            return render_template('page2.html', issuers=issuers, error="Invalid issuer selected.")

        graph = generateGraph(issuer, df)

        return render_template('page2.html', issuers=issuers, graph=graph, selected_issuer=issuer)

    else:
        return render_template('page2.html', issuers=issuers)


@app.route('/page3', methods=['GET', 'POST'])
def page3():
    return render_template('page3.html', issuers=issuers)


if __name__ == '__main__':
    app.run(debug=True)
