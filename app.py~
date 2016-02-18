from flask import Flask, render_template, request, redirect
#import csv
import pandas as pd


from bokeh.plotting import figure
from bokeh.embed import components 
from bokeh.charts import Bar

app = Flask(__name__)

@app.route('/')
def main():
  return redirect('/index')

@app.route('/index')
def index():
  return render_template('index.html')

@app.route('/graph1')
def graph1():

#  myData=pd.read_csv('myData.csv')
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
  predicted=[180.037,199.848,371.858,431.991,469.749,496.32,571.836,601.203,606.797,619.383,634.766,718.673,724.267,728.462,788.596,813.768,819.362,830.549,831.948,889.284,924.245,928.441,952.214,1023.53,1047.31,1059.89,1103.25,1160.58,1219.32]

  plot.line(x,predicted,color='black',line_width=3)

  script, div = components(plot)
  return render_template('graph.html', script=script, div=div)

@app.route('/graph2')
def graph2():

  towns=["Yorktown","Croton-on-Hudson","Ardsley","Ossining","Chappaqua","Lewisboro","Elmsford","Somers","Pleasantville","North Salem","Mount Vernon","New Rochelle","Armonk","White Plains","Dobbs Ferry","Greenburgh","Peekskill","Port Chester","Bedford","Mount Pleasant","Eastchester","Mamaroneck","Scarsdale","Irvington","Tuckahoe","Bronxville","Rye","Harrison","Yonkers"]

  pctDiff=[-44.78,-42.63,-41.59,-41.24,-31.13,-28.63,-25.64,-25.37,-23.48,-18.64,-16.18,-8.03,-7.70,-7.45,-5.31,-2.07,-2.00,4.01,6.68,8.54,10.71,11.44,14.30,18.74,37.19,57.59,68.30,92.50,100.59]

  df = pd.DataFrame({ 'Municipalities' : towns,'pctDiff' : pctDiff})
  plot = Bar(df, label='Municipalities', values='pctDiff', title="Under-/Over-valued municipalities",agg='min',color='green')

  script, div = components(plot)
  return render_template('graph.html', script=script, div=div)

#predHousePrice_bySAT

if __name__ == '__main__':
  app.run(port=33507)
#  app.run(debug=True)


