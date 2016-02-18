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
  plot = figure(width=450, height=450, title='Median Home Prices vs. SAT Score',
                x_axis_label='SAT Score')
  plot.circle(myData['Mean Total SAT'],myData['Median Home Sale Price'], size=10)
  #plot.circle([1,2,3],[1,2,3])
  
  x=myData['Mean Total SAT'].tolist()
  y=pd.read_csv('predHousePrice_bySAT.csv')
  y1=y['predicted_price'].tolist()
  xy = zip(x,y1)
  xy_sorted = sorted(xy, key = lambda x : x[0],reverse=False)
  
  plot.line(list(zip(*xy_sorted)[0]),list(zip(*xy_sorted)[1]),color='black',line_width=3)

  script, div = components(plot)
  return render_template('graph.html', script=script, div=div)


#predHousePrice_bySAT

if __name__ == '__main__':
  app.run(port=33507)
#  app.run(debug=True)
