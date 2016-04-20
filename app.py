from __future__ import division
from flask import Flask, render_template, request, redirect

import os
import psycopg2
import urlparse

import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer

from bokeh.plotting import *
from bokeh.embed import components
from bokeh.models import HoverTool,sources
from collections import OrderedDict

app = Flask(__name__)

# global variables to be used on different pages
app.vars={}
app.selectedids=[]
app.validids=[]
app.refnames=[]
app.prices_actual=[]
app.prices_predicted=[]
app.features=[]
app.locscores=[]
app.zlat=0
app.zlong=0
app.feature_string=[]

@app.route('/')
def main():
  return redirect('/index')

@app.route('/index', methods=['POST', 'GET'])
def index():

  if request.method == 'POST':

    # get user-defined search area (currently not used)
    app.vars['qcenter'] = int(request.form['center'])
    app.vars['qradius']= float(request.form['radius'].replace(' miles',''))
    app.vars['qfeatures'] = request.form.getlist('features')
    app.feature_string =  getFeatureString(app.vars['qfeatures'])
    
    # get data from postgresql
    urlparse.uses_netloc.append("postgres")
    url = urlparse.urlparse(os.environ["DATABASE_URL"])
    try:
      #conn = psycopg2.connect("dbname='nysRealEstate' user='enghuiy' host='localhost' password=''")
      conn = psycopg2.connect(database=url.path[1:],user=url.username,password=url.password,host=url.hostname,port=url.port)
    except:
      return "Error: unable to connect to database"

    cur = conn.cursor()
    # 1) get long/lat base on zipcode
    cur.execute("""select lat,long from zipcode2longlat where zipcode=%d;""" % app.vars['qcenter'])
    app.zlat,app.zlong=cur.fetchone()

    # 2) get refshp_indices with (long/lat - centroid).dist < radius
    cur.execute("""SELECT refshpindex,centroid_lat,centroid_long FROM refshape;""")
    id_centroids=cur.fetchall()
    app.selectedids=[]
    for refshpindex,clat,clong in id_centroids: 
      d=distance_on_unit_sphere(clat, clong, app.zlat, app.zlong)
      if d <=app.vars['qradius']:
        app.selectedids.append(refshpindex)

    # 3) select price,features from lines matching these refshp_indices
#    cur.execute("""SELECT refshape.name,zprice,sch_perform FROM price2features JOIN refshape ON price2features.refshpindex=refshape.refshpindex WHERE sch_perform > 0;""")
    queryString = generateQueryString_priceFeatures(app.vars['qfeatures'],app.selectedids)
    #print queryString
    cur.execute(queryString)
    
    data=zip(*cur.fetchall())
    app.validids = list(data[0])
    app.refnames = list(data[1])
    app.prices_actual = data[2]
    nrows=len(data[3])
    nfeatures=len(data[3:])
    app.features  = np.asarray(data[3:]).reshape(nrows,nfeatures)
    
    #==========================================================
    # MACHINE LEARNING

    # fill in zero values
    app.features = fillNAs(app.features)
    
    # run linear regression
    coeffs,intercept,r2,app.prices_predicted =  linearRegression(app.features,app.prices_actual)
    app.locscores = [ (app.prices_actual[i]-app.prices_predicted[i])/app.prices_predicted[i]*100 for i in range(len(app.prices_predicted))]

    # write out fit stats
    #==========================================================
    # plot with bokeh
    #script, div = plotLR(features,homevalue_scaled,ypredicted_scaled,refnames,coeffs[featureIndex],intercept,r2)
    #script, div = plotPrice2PriceLR(app.features,app.prices_actual,app.prices_predicted,app.refnames,coeffs,intercept,r2)
    #coeff_string=''
    #return render_template('graph.html', script=script, div=div, featureString=app.feature_string,coeffString=coeff_string,intercept='%d'%intercept,r2='%4.2f'%r2)
    #return render_template('temp.html',data=homevalue[0])
    #==========================================================
    # plot map

      #get the shapejsons from postgresql
    try:
      queryString=generateQueryString_json(app.validids)
      print queryString
      cur.execute(queryString)
    except:
      print "cannot get jsons from database"

    r = cur.fetchall()
    geojsonFeatures = zip(*r)[0]
    temp=[] 
    for i,jsonstring in enumerate(geojsonFeatures):
      t1=jsonstring.replace('"STATEFP": "36"','"score":%f, "actual_price":%f, "predicted_price":%f' %(app.locscores[i],app.prices_actual[i],app.prices_predicted[i]))
      t2=t1.replace('{"type": "FeatureCollection", "features": [', '')
      t3=t2.replace('}}]}', '}}')
      temp.append(t3)
    
    geojsonFeatures_new='{"type": "FeatureCollection", "features": ['+','.join(temp)+']}'
    return render_template('map_test.html', featureString=app.feature_string,gjson=geojsonFeatures_new,center_lat=app.zlat,center_long=app.zlong)

  return render_template('index.html')

@app.route('/map')
def map_test():

  # return to index page if there are no data
  if not app.validids:
    return render_template('index.html')

  urlparse.uses_netloc.append("postgres")
  url = urlparse.urlparse(os.environ["DATABASE_URL"])
  try:
    #conn = psycopg2.connect("dbname='nysRealEstate' user='enghuiy' host='localhost' password=''")
    conn = psycopg2.connect(database=url.path[1:],user=url.username,password=url.password,host=url.hostname,port=url.port)
  except:
    return "Error: unable to connect to database"

  cur = conn.cursor()

  #get the shapejsons from postgresql
  try:
    queryString=generateQueryString_json(app.validids)
    print queryString
    cur.execute(queryString)
  except:
    print "cannot get jsons from database"

  r = cur.fetchall()
  geojsonFeatures = zip(*r)[0]
  temp=[] 
  for i,jsonstring in enumerate(geojsonFeatures):
    t1=jsonstring.replace('"STATEFP": "36"','"score":%f, "actual_price":%f, "predicted_price":%f' %(app.locscores[i],app.prices_actual[i],app.prices_predicted[i]))
    t2=t1.replace('{"type": "FeatureCollection", "features": [', '')
    t3=t2.replace('}}]}', '}}')
    temp.append(t3)
    
  geojsonFeatures_new='{"type": "FeatureCollection", "features": ['+','.join(temp)+']}'

  
  return render_template('map_test.html', featureString=app.feature_string,gjson=geojsonFeatures_new,center_lat=app.zlat,center_long=app.zlong)

@app.route('/method')
def method():
  return render_template('method.html')

@app.route('/plots')
def plots():
  return render_template('plots.html')

#===================================================
 
def distance_on_unit_sphere(lat1, long1, lat2, long2):
 
    # Convert latitude and longitude to 
    # spherical coordinates in radians.
    degrees_to_radians = math.pi/180.0
         
    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians
         
    # theta = longitude
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians
         
    # Compute spherical distance from spherical coordinates.
    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) +
           math.cos(phi1)*math.cos(phi2))
    arc = math.acos( cos )
 
    # multiply arc by the radius of the earth in desired unit.
    return arc*3960 # miles
    #return arc*6373 # km

# MACHINE LEARNING
def fillNAs(Xtrain_in):
  imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
  return imp.fit_transform(Xtrain_in)

def linearRegression(features,prices_actual):
    lin_reg = LinearRegression(normalize=True)
    lin_reg.fit(features,prices_actual)

    prices_predicted = lin_reg.predict(features)
    coeffs = lin_reg.coef_
    intercept = lin_reg.intercept_
    r2=lin_reg.score(features,prices_actual)

    return (coeffs,intercept,r2,prices_predicted)

# PLOTTING WITH BOKEH
def mtext(p, x, y, text):
    p.text(x, y, text=[text],
           text_color="#525252",text_align="left", text_font_size="11pt")

def plotAlternatives(prices_actual,homevalue,y_predicted,refnames,coefficient,intercept,r2):
  TOOLS = 'box_zoom,box_select,resize,reset,hover'
  
  plot = figure(width=600, height=400,y_axis_label='Prices', x_axis_label='Locales',tools=TOOLS)
  plot.line(feature1D,y_predicted,color='black',line_width=3)

  source = ColumnDataSource(
    data=dict(
      x=feature1D,
      y=homevalue,
      label=refnames
      )
    )

  plot.circle('x', 'y', color='blue',size=7, alpha=0.7, source=source)

  hover =plot.select(dict(type=HoverTool))
  hover.tooltips = OrderedDict([
    ("Locale", "@label"),
    ("(feature,price)", "(@x, @y)"),
    ])

  mtext(plot, min(feature1D) + 1,max(homevalue)-100000, "y = %6.2f x + (%6.2f)" %(coefficient,intercept))
  mtext(plot, min(feature1D) + 1,max(homevalue)-250000, "R2 = %5.3f" %(r2))

  script, div = components(plot)

  return (script, div)

def plotLR(feature1D,homevalue,y_predicted,refnames,coefficient,intercept,r2):
  TOOLS = 'box_zoom,box_select,resize,reset,hover'
  
  plot = figure(width=600, height=400,y_axis_label='Home Price ($)', x_axis_label='Features',tools=TOOLS)
  plot.line(feature1D,y_predicted,color='black',line_width=3)

  source = ColumnDataSource(
    data=dict(
      x=feature1D,
      y=homevalue,
      label=refnames
      )
    )

  plot.circle('x', 'y', color='blue',size=7, alpha=0.7, source=source)

  hover =plot.select(dict(type=HoverTool))
  hover.tooltips = OrderedDict([
    ("Locale", "@label"),
    ("(feature,price)", "(@x, @y)"),
    ])

  mtext(plot, min(feature1D) + 1,max(homevalue)-100000, "y = %6.2f x + (%6.2f)" %(coefficient,intercept))
  mtext(plot, min(feature1D) + 1,max(homevalue)-250000, "R2 = %5.3f" %(r2))

  script, div = components(plot)

  return (script, div)

def plotPrice2PriceLR(features,prices_actual,prices_predicted,refnames,coefficients,intercept,r2):

  scaleFactor = 1000
  prices_predicted_scaled = [ x / scaleFactor for x in prices_predicted]
  prices_actual_scaled = [ x / scaleFactor for x in prices_actual]


  TOOLS = 'box_zoom,box_select,resize,reset,hover'
  mina = min(prices_actual_scaled)-50
  maxa = max(prices_actual_scaled)+50

  minp = min(prices_predicted_scaled)-50
  maxp = max(prices_predicted_scaled)+50

  one2one = np.linspace(minp,maxp,20)
  
  plot = figure(width=600, height=450,
                x_axis_type="log", x_range=[minp,maxp],
                y_axis_type="log", y_range=[mina,maxa],
                y_axis_label='Actual Home Price (Thousands $)',
                x_axis_label='Expected Home Price (Thousands $)',tools=TOOLS)

  plot.line(one2one,one2one,color='#525252',line_width=2,line_dash=[4,4])
  plot.xaxis.major_label_orientation = 3.142/4

  source = ColumnDataSource(
    data=dict(
      #x=feature1D,
      y0=prices_actual_scaled,
      y1=prices_predicted_scaled,
      label=refnames
      )
    )

  plot.circle('y1', 'y0', color='#4575b4',size=7, alpha=0.7, source=source)

  hover =plot.select(dict(type=HoverTool))
  hover.tooltips = OrderedDict([
    ("Locale", "@label"),
    ("Actual ($K)", "@y0"),
    ("Expected ($K)", "@y1"),
    ])

#  mtext(plot, min(y_predicted) + 2000,max(homevalue)-100000, "y = %6.2f x + (%6.2f)" %(coefficient,intercept))
#  mtext(plot, min(y_predicted) + 2000,max(homevalue)-250000, "R2 = %5.3f" %(r2))

  script, div = components(plot)

  return (script, div)

def getFeatureString(featurelist):
  outstring=''
  for i,f in enumerate(featurelist):
    if f=='school_performance':
      outstring += 'school performance'
    elif f=='crime_rate':
      outstring+='crime rate'
    elif f=='commute_time':
      outstring+='commute time'
    elif f=='walkability':
      outstring+='walkability'
    elif f=='roi':
      outstring+='ROI'
    else:
      pass
    if i<len(featurelist)-1:
      outstring+=', '
  return outstring

# generate the query to get features
def generateQueryString_priceFeatures(featurelist,refshpindexlist):
    featurestring=''
    for i,f in enumerate(featurelist):
        if f=='school_performance':
            featurestring += 'sch_perform'
        elif f=='crime_rate':
            featurestring+='crimerate_total'
        elif f=='commute_time':
            featurestring+='commute_time'
        elif f=='walkability':
            featurestring+='walkability'
        elif f=='roi':
            featurestring+='roi'
        else:
            pass
        if i<len(featurelist)-1:
            featurestring+=','
        
    criteriastring='refshape.refshpindex in ('+ ','.join([str(i) for i in refshpindexlist])+')'

    outstring='SELECT refshape.refshpindex,refshape.name,zprice,'+ featurestring + ' from price2features JOIN refshape ON price2features.refshpindex=refshape.refshpindex WHERE '+ criteriastring + ';'
        
    return outstring

# generate the query to get features
def generateQueryString_json(refshpindexlist):
    criteriastring='refshpindex in ('+ ','.join([str(i) for i in refshpindexlist])+')'
    outstring='SELECT json FROM refshape WHERE '+ criteriastring + ';'
        
    return outstring


def generateHTML_fitstats(coeffs,intercept,r2):
  outstring='Baseline price = $%dK<br>' % intercept/1000
  for i,f in enumerate(featurelist):
    if f=='school_performance':
      featurestring += 'school performance: '
    elif f=='crime_rate':
      featurestring+='crimerate_total'
    elif f=='commute_time':
      featurestring+='commute_time'
    elif f=='walkability':
      featurestring+='walkability'
    elif f=='roi':
      featurestring+='roi'
    else:
      pass
    if i<len(featurelist)-1:
      featurestring+=','


# RUN
if __name__ == '__main__':
#  app.run(port=33507)
  app.run(debug=True,port=5001)
