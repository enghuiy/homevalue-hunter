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
  plot = figure(width=450, height=450, y_axis_label='Median Home Sale Price (thousands)',
                x_axis_label='SAT Score')
  
  x=myData['Mean Total SAT'].tolist()
#  actual=myData['Median Home Sale Price'].tolist()
#  actual_th=[aa/1000 for aa in actual]
  actual_th=[435,390,255.25,433,364.5,478,640,618,680,390,527,1300,690,528.25,422.5,749,831,570,535,1035,550,991,627,1835,627.5,1800,1056,892.5,1497]

  plot.circle(x,actual_th, size=10)

  y=pd.read_csv('predHousePrice_bySAT.csv')
  y1=y['predicted_price'].tolist()
  y_th=[yy/1000 for yy in y1]
  xy = zip(x,y_th)
  xy_sorted = sorted(xy, key = lambda x : x[0],reverse=False)
  
  plot.line(list(zip(*xy_sorted)[0]),list(zip(*xy_sorted)[1]),color='black',line_width=3)

  script, div = components(plot)
  return render_template('graph.html', script=script, div=div)


#predHousePrice_bySAT

if __name__ == '__main__':
  app.run(port=33507)
#  app.run(debug=True)


