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
  
#  x=myData['Mean Total SAT'].tolist()
#  actual=myData['Median Home Sale Price'].tolist()
#  actual_th=[aa/1000 for aa in actual]
  x=[1191.8,1206.0,1329.0,1372.0,1399.0,1418.0,1472.0,1493.0,1497.0,1506.0,1517.0,1577.0,1581.0,1584.0,1627.0,1645.0,1649.0,1657.0,1658.0,1699.0,1724.0,1727.0,1744.0,1795.0,1812.0,1821.0,1852.0,1893.0,1935.0]
  actual_th=[435,390,255.25,433,364.5,478,640,618,680,390,527,1300,690,528.25,422.5,749,831,570,535,1035,550,991,627,1835,627.5,1800,1056,892.5,1497]

  plot.circle(x,actual_th, size=10)

#  y=pd.read_csv('predHousePrice_bySAT.csv')
#  y1=y['predicted_price'].tolist()
#  y_th=[yy/1000 for yy in y1]
#  xy = zip(x,y_th)
#  xy_sorted = sorted(xy, key = lambda x : x[0],reverse=False)  
#  plot.line(list(zip(*xy_sorted)[0]),list(zip(*xy_sorted)[1]),color='black',line_width=3)
  predicted=[216.858,465.26,260.452,416.305,490.198,488.091,695.914,667.78,495.654,663.705,485.537,675.315,623.262,649.261,765.09,791.005,778.975,744.91,716.888,928.786,958.664,834.573,878.505,1090.35,1074.24,1142.23,1144.06,1295.92,1309.7]
  plot.line(x,predicted,color='black',line_width=3)

  script, div = components(plot)
  return render_template('graph.html', script=script, div=div)


#predHousePrice_bySAT

if __name__ == '__main__':
  app.run(port=33507)
#  app.run(debug=True)


