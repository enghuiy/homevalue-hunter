from flask import Flask, render_template, request, redirect

import os
import psycopg2
import urlparse

import numpy as np
from sklearn import linear_model as lm
from sklearn.metrics import r2_score
#import pandas as pd

#from bokeh.plotting import figure
#from bokeh.embed import components
#from bokeh.models import HoverTool

app = Flask(__name__)

@app.route('/')
def main():
  return redirect('/index')

@app.route('/index', methods=['POST', 'GET'])
def index():

  if request.method == 'POST':

    # get user-defined search area (currently not used)
    query_center=request.form['center']
    query_radius=request.form['radius']
    query_features=request.form['features']
    # blah blah

    # get data from postgresql
    urlparse.uses_netloc.append("postgres")
    url = urlparse.urlparse(os.environ["DATABASE_URL"])
    try:
      #conn = psycopg2.connect("dbname='nysRealEstate' user='enghuiy' host='localhost' password=''")
      conn = psycopg2.connect(database=url.path[1:],user=url.username,password=url.password,host=url.hostname,port=url.port)

    except:
      print "I am unable to connect to the database"

    cur = conn.cursor()
    cur.execute("""SELECT "ZPRICE","SCORE" from price2features WHERE "SCORE" > 0""")
    data=zip(*cur.fetchall())
    homevalue = list(data[0])
    features  = list(data[1])
    #print "%10f %4.1f" %(homevalue[0],features[0]) 

    # run linear regression

    #r2,ypredicted =  linearRegression(features,homevalue)
    #ypredicted_scaled = [ x / 1000 for x in ypredicted]
    #homevalue_scaled = [ x / 1000 for x in homevalue]

    # plot with bokeh
    #plot = figure(width=450, height=450, y_axis_label='Home Price', x_axis_label='Features')
    #y=[0,1,2]
    #plot.line(y,y,color='green',line_width=2)
    #script, div = components(plot)
    #script, div = plotLR(features,homevalue_scaled,ypredicted_scaled)

    #return render_template('graph.html', script=script, div=div)
    return render_template('temp.html',data=homevalue[0])

  return render_template('index.html')

#===================================================
# normalization
def norm(x_in,x_norm):
    
    x_mu = np.mean(x_in)
    x_range = np.amax(x_in) - np.amin(x_in)
    x_norm [:] = [ ( x - x_mu ) / float (x_range) for x in x_in]
    return (x_mu, x_range)

# convert back to abs value
def unnorm(x_mu, x_range, x_norm):
    x_out=[]
    x_out [:] = [ x*x_range+x_mu for x in x_norm ]
    return x_out

# univariate regression
# univariate regression
def linearRegression(features,homevalue):
    x_norm=[]; y_norm=[]
    (x_mu,x_range) = norm(features,x_norm)
    (y_mu, y_range) = norm(homevalue, y_norm)

    X_train = np.asarray(zip( np.ones(len(x_norm)),x_norm))

    # Create linear regression object
    regr = lm.LinearRegression()
    regr.fit(X_train, y_norm)

    # convert y back to abs value
    y_predicted_norm = regr.predict(X_train)
    y_predicted = unnorm(y_mu,y_range,y_predicted_norm)

    r2=r2_score(y_norm, y_predicted_norm)

    return (r2,y_predicted)

# plot with bokeh
def plotLR(feature1D,homevalue,y_predicted):

  TOOLS = 'box_zoom,box_select,resize,reset,hover'
  
  plot = figure(width=500, height=500, y_axis_label='Home Price (thousands)', x_axis_label='Features',tools=TOOLS)
  plot.line(feature1D,y_predicted,color='green',line_width=3)
  plot.circle(feature1D,homevalue, color='grey',size=3)

  script, div = components(plot)

  return (script, div)

# RUN
if __name__ == '__main__':
#  app.run(port=33507)
  app.run(debug=True)
