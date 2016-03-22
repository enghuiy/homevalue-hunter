from flask import Flask, render_template, request, redirect

import os
import psycopg2
import urlparse

import numpy as np
from sklearn import linear_model as lm
from sklearn.metrics import r2_score
#import pandas as pd

from bokeh.plotting import *
from bokeh.embed import components
from bokeh.models import HoverTool,sources
from collections import OrderedDict

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
    feature_string = query_features
    
    # blah blah

    # get data from postgresql
    urlparse.uses_netloc.append("postgres")
    url = urlparse.urlparse(os.environ["DATABASE_URL"])
    try:
      #conn = psycopg2.connect("dbname='nysRealEstate' user='enghuiy' host='localhost' password=''")
      conn = psycopg2.connect(database=url.path[1:],user=url.username,password=url.password,host=url.hostname,port=url.port)

    except:
      return "Error: unable to connect to database"

    cur = conn.cursor()
    cur.execute("""SELECT refshape."NAME","ZPRICE","SCORE" from price2features JOIN refshape ON  price2features."refSHPindex"=refshape."refSHPindex" WHERE "SCORE" > 0;""")
    data=zip(*cur.fetchall())
    refnames = list(data[0])
    homevalue = list(data[1])
    features  = list(data[2])
    #print "%10f %4.1f" %(homevalue[0],features[0]) 

    # run linear regression

    coeffs,intercept,r2,ypredicted =  linearRegression(features,homevalue)
    print coeffs
    print intercept
    ypredicted_scaled = [ x / 1 for x in ypredicted]
    homevalue_scaled = [ x / 1 for x in homevalue]

    # plot with bokeh
    #plot = figure(width=450, height=450, y_axis_label='Home Price', x_axis_label='Features')
    #plot.line(y,y,color='green',line_width=2)
    featureIndex=1
    script, div = plotLR(features,homevalue_scaled,ypredicted_scaled,refnames,coeffs[featureIndex],intercept,r2)

    return render_template('graph.html', script=script, div=div, features=feature_string)
    #return render_template('temp.html',data=homevalue[0])

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
def linearRegression(features,homevalue):
    X_train = np.asarray(zip( np.ones(len(features)),features))

    # Create linear regression object
    regr = lm.LinearRegression()
    regr.fit(X_train, homevalue)
    coeffs = regr.coef_
    intercept = regr.intercept_

    # convert y back to abs value
    y_predicted_norm = regr.predict(X_train)
    y_predicted = regr.predict(X_train)

    r2=r2_score(homevalue, y_predicted)

    return (coeffs,intercept,r2,y_predicted)

def linearRegression_mynorm(features,homevalue):
    x_norm=[]; y_norm=[]
    (x_mu,x_range) = norm(features,x_norm)
    (y_mu, y_range) = norm(homevalue, y_norm)

    X_train = np.asarray(zip( np.ones(len(x_norm)),x_norm))

    # Create linear regression object
    regr = lm.LinearRegression()
    regr.fit(X_train, y_norm)
    coeffs = regr.coef_
    intercept = regr.intercept_

    # convert y back to abs value
    y_predicted_norm = regr.predict(X_train)
    y_predicted = unnorm(y_mu,y_range,y_predicted_norm)

    r2=r2_score(y_norm, y_predicted_norm)

    return (coeffs,intercept,r2,y_predicted)


# plot with bokeh
def mtext(p, x, y, text):
    p.text(x, y, text=[text],
           text_color="green", text_font_style='bold',text_align="left", text_font_size="14pt")

def plotLR(feature1D,homevalue,y_predicted,refnames,coefficient,intercept,r2):

  TOOLS = 'box_zoom,box_select,resize,reset,hover'
  
  plot = figure(width=600, height=450,y_axis_label='Home Price ($)', x_axis_label='Features',tools=TOOLS)
  plot.line(feature1D,y_predicted,color='green',line_width=3)

  source = ColumnDataSource(
    data=dict(
      x=feature1D,
      y=homevalue,
      label=refnames
      )
    )

  plot.circle('x', 'y', color='grey',size=5, alpha=0.5, source=source)

  hover =plot.select(dict(type=HoverTool))
  hover.tooltips = OrderedDict([
    ("Locale", "@label"),
    ("(feature,price)", "(@x, @y)"),
    ])

  mtext(plot, min(feature1D) + 1,max(homevalue)-100000, "y = %6.2f x + (%6.2f)" %(coefficient,intercept))
  mtext(plot, min(feature1D) + 1,max(homevalue)-250000, "R2=%5.3f" %(r2))

  script, div = components(plot)

  return (script, div)

# RUN
if __name__ == '__main__':
#  app.run(port=33507)
  app.run(debug=True)
