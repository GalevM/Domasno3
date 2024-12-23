import pandas as pd
from flask import Flask, render_template
import matplotlib

matplotlib.use('Agg')

app = Flask("ScrapingDocs")

dataFile = pd.read_csv("../../Domasno1/dokss.csv")

issuers = [row for row in dataFile['Ime na Kompanija']]
issuers = set(issuers)
issuers = sorted(issuers)



@app.route('/')
def index():
    """Render the homepage."""
    return render_template('page1.html')


@app.route('/page2', methods=['GET', 'POST'])
def show_graphs():
    return render_template('page2.html', issuers=issuers)


@app.route('/page3', methods=['GET', 'POST'])
def page3():
    return render_template('page3.html', issuers=issuers)


if __name__ == '__main__':
    app.run(debug=True)
