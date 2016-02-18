from flask import Flask, render_template, request, redirect
import csv
import pandas as pd

from bokeh.plotting import figure
from bokeh.embed import components 

app = Flask(__name__)

@app.route('/')
def main():
  return redirect('/index')

@app.route('/index')
def index():
  return render_template('index.html')

@app.route('/graph')
def graph():

  myData=pd.read_csv('myData.csv')
  plot = figure(title='Median Home Prices vs. SAT Score',
                x_axis_label='SAT Score')
  plot.circle(myData['Mean Total SAT'],myData['Median Home Sale Price'],)
  
  script, div = components(plot)
  return render_template('graph.html', script=script, div=div)




#predHousePrice_bySAT

if __name__ == '__main__':
  app.run(port=33507)
