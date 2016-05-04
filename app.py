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

#from wordcloud import WordCloud
import matplotlib.pyplot as plt
import mpld3
import ujson
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
app.locale_names_json=''
@app.route('/')
def main():
  return redirect('/index')

@app.route('/index', methods=['POST', 'GET'])
def index():

  if request.method == 'POST':

    # get data from postgresql
    try:
      urlparse.uses_netloc.append("postgres")
      url = urlparse.urlparse(os.environ["DATABASE_URL"])
      try:
        conn = psycopg2.connect(database=url.path[1:],user=url.username,password=url.password,host=url.hostname,port=url.port)
      except:
        return "Error: unable to connect to database"
    except:
      try:
        conn = psycopg2.connect("dbname='nysRealEstate' user='enghuiy' host='localhost' password=''")
      except:
        return "Error: unable to connect to database"
      
    cur = conn.cursor()

    # get user-defined search area
    center_string=request.form['center']
    try:
      # query is zipcode; get long/lat base from zipcode
      center_zipcode = int(center_string)
      cur.execute("""select lat,long from zipcode2longlat where zipcode=%d;""" % center_zipcode)
      data = cur.fetchone()
      if  data == None:
        return render_template('index.html',msg='zipcode %d not found' % center_zipcode)
      app.zlat,app.zlong=data

    except:
      # query is city name; get long/lat base from locale_name
      center_localename = center_string # query is city
      cur.execute("""SELECT centroid_lat,centroid_long FROM refshape WHERE locale_name='%s';""" % center_localename)
      data = cur.fetchone()
      if  data == None:
        return render_template('index.html',msg='city name %s not found' % center_localename)
      app.zlat,app.zlong=data

    app.vars['qradius']= float(request.form['radius'].replace(' miles',''))
    app.vars['qfeatures'] = request.form.getlist('features')
    app.feature_string =  getFeatureString(app.vars['qfeatures'])
    
    # 2) get refshp_indices with (long/lat - centroid).dist < radius
    cur.execute("""SELECT refshpindex,centroid_lat,centroid_long FROM refshape;""")
    id_centroids=cur.fetchall()
      
    app.selectedids=[]
    for refshpindex,clat,clong in id_centroids: 
      d=distance_on_unit_sphere(clat, clong, app.zlat, app.zlong)
      if d <=app.vars['qradius']:
        app.selectedids.append(refshpindex)

    if not app.selectedids:
      return render_template('index.html',msg='No such locales found')

    # 3) select price,features from lines matching these refshp_indices
#    cur.execute("""SELECT refshape.name,zprice,sch_perform FROM price2features JOIN refshape ON price2features.refshpindex=refshape.refshpindex WHERE sch_perform > 0;""")
    queryString = generateQueryString_priceFeatures(app.vars['qfeatures'],app.selectedids)
    #print queryString
    cur.execute(queryString)
    
    data=zip(*cur.fetchall())
    if not data:
      return render_template('index.html',msg='No qualifying locales found in database.')
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

  # GET METHOD
      # get data from postgresql
  try:
    urlparse.uses_netloc.append("postgres")
    url = urlparse.urlparse(os.environ["DATABASE_URL"])
    try:
      conn = psycopg2.connect(database=url.path[1:],user=url.username,password=url.password,host=url.hostname,port=url.port)
    except:
      return "Error: unable to connect to database"
  except:
    try:
      conn = psycopg2.connect("dbname='nysRealEstate' user='enghuiy' host='localhost' password=''")
    except:
      return "Error: unable to connect to database"
  cur = conn.cursor()
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  #get user-defined search area
  cur.execute("""select locale_name from refshape;""") # add state later
  data = cur.fetchall()
  app.locale_names_json=ujson.dumps({'locale_names':zip(*data)[0]})
  return render_template('index.html', json=app.locale_names_json )



@app.route('/map')
def map_test():

  # return to index page if there are no data
  if not app.validids:
    return redirect('/index')

    try:
      urlparse.uses_netloc.append("postgres")
      url = urlparse.urlparse(os.environ["DATABASE_URL"])
      try:
        conn = psycopg2.connect(database=url.path[1:],user=url.username,password=url.password,host=url.hostname,port=url.port)
      except:
        return "Error: unable to connect to database"
    except:
      try:
        conn = psycopg2.connect("dbname='nysRealEstate' user='enghuiy' host='localhost' password=''")
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

@app.route('/ahead')
def ahead():
  return render_template('ahead.html')

@app.route('/info/<int:refid>')
def info(refid):
  
    # get locale data from postgresql
    try:
      urlparse.uses_netloc.append("postgres")
      url = urlparse.urlparse(os.environ["DATABASE_URL"])
      try:
        conn = psycopg2.connect(database=url.path[1:],user=url.username,password=url.password,host=url.hostname,port=url.port)
      except:
        return "Error: unable to connect to database"
    except:
      try:
        conn = psycopg2.connect("dbname='nysRealEstate' user='enghuiy' host='localhost' password=''")
      except:
        return "Error: unable to connect to database"
    cur = conn.cursor()

    # get name
    queryString='SELECT name from refshape WHERE refshpindex= %d;' % refid
    cur.execute(queryString)
    data = cur.fetchone()
    if  data == None:
      return render_template('locale_info.html',msg='No locale info found')
    localename=data[0]

    # get price+features
    featurestring='sch_perform,crimerate_total,roi,traveltime,walkability,text'
    queryString='SELECT zprice,'+ featurestring +' from price2features WHERE refshpindex= %d;' % refid
    cur.execute(queryString)    
    data = cur.fetchone()
    if  data == None:
      return render_template('locale_info.html',msg='No locale info found')
    medhomeprice="%d" % data[0] if data[0] else 'no data'
    school="%.1d" % data[1] if data[1] else 'no data'
    crime="%.2f" % data[2] if data[2] else 'no data'
    roi="%.2f" % data[3] if data[3] else 'no data'
    traveltime="%.2f" % data[4] if data[4] else 'no data'
    walkability="%d" % data[5] if data[5] else 'no data'
    
    keytext=data[6] if data[6] else ''
    keytext = ''
    
    # wordcloud
    if keytext:
      fig, ax = plt.subplots()
      wordcloud = WordCloud(max_font_size=40, relative_scaling=.5).generate(keytext) #numpy array
      wordcloud=np.flipud(wordcloud)
      ax.imshow(wordcloud)
      ax.set_title('What residents say about the locale')
      fig.tight_layout()
      #ax.scatter([1,2],[1,2])

      mpld3.plugins.clear(fig)  # clear all plugins from the figure
      cloudjson=mpld3.fig_to_dict(fig)
      cloudjson['width']=533
      cloudjson['height']=400
      cloudjson=ujson.dumps(cloudjson)
    else:
      cloudjson=''
      
    return render_template('locale_info.html',msg='',name=localename,medhomeprice=medhomeprice,school=school,crime=crime,roi=roi,traveltime=traveltime,walkability=walkability,cloudjson=cloudjson)
    #return render_template('test.html',json=cloud_json)

@app.route('/alternatives')
def alternatives():
  refnames=['Yonkers','Greenburg','White Plains','New Rochelle','Eastchester','Bronxville','Edgemont','Scarsdale']
  prices_actual=[407.975,410.360,501.440,564.475,609.400,678.300,953.500,1452.200]
  good_features=[0,0,0,0,1,1,1,1]
  
  minhouseprice = 407.925 
  cost_elemschool = 5.845
  cost_highschool = 22.477
  cost_perchild = 9*cost_elemschool + 4*cost_highschool
  maxchild = 5
  prices_alternate1 = [ minhouseprice+cost_perchild*i for i in range(1,maxchild+1)]

  prices_alternateStep = sorted([minhouseprice,minhouseprice]+prices_alternate1+prices_alternate1[:-1])
  childlistStep=[0,1,1,2,2,3,3,4,4,5]
  
  nLocales = len(refnames)
  
  TOOLS = ''

  sorted_label_prices = sorted(zip(refnames,prices_actual),key=lambda x:x[1])
  t=zip(*sorted_label_prices)

  plot = figure(width=600, height=400,y_axis_label='Home Price ($ thousands)', x_axis_label='No. of children',tools=TOOLS)
  plot.line(childlistStep,prices_alternateStep,color='black',line_width=3)

  plot.circle(0,t[1][0], color='orange',size=15, alpha=1)
  mtext(plot,0,(t[1][0]+60), "cheapest")
  mtext(plot,0,(t[1][0]+20), "locale")

  plot.line([0,maxchild],[t[1][4],t[1][4]], color='blue',line_width=3)
  mtext(plot, 2.5,(t[1][4]+1), "cheapest 'good school' locale")


  plot1 = figure(width=600, height=400,y_axis_label='Home Price ($ thousands)', x_axis_label='Locales',tools=TOOLS)
  source1 = ColumnDataSource(data=dict(label=t[0],x=range(nLocales),ay=t[1]))

  plot1.circle(range(4,8), t[1][4:8], color='blue',size=15, alpha=1, legend="Locales with good schools")
  plot1.circle(range(0,4), t[1][0:4], color='orange',size=15, alpha=1,legend="Locales with bad schools")
  plot1.legend.orientation = "top_left"
  
#  hover = plot.select(dict(type=HoverTool))
#  hover.tooltips = OrderedDict([("Locale ", "@label"),("Price ", "@ay")])

  plot2 = figure(width=600, height=400,y_axis_label='Home Price ($ thousands)', x_axis_label='Locales',tools=TOOLS)
  for i in range(maxchild):
    plot2.line([0,nLocales],[prices_alternate1[i],prices_alternate1[i]],color='blue',line_dash=[10,10],line_width=1)
    if i==0:
      mtext(plot2, 0,(prices_alternate1[i]+1), "house + %d tuition" % (i+1))
    else:
      mtext(plot2, 0,(prices_alternate1[i]+1), "house + %d tuitions" % (i+1))


  plot2.circle(range(0,4), t[1][0:4], color='orange',size=15, alpha=1)
  plot2.circle(range(4,8), t[1][4:8], color='blue',size=15, alpha=1)

  #script, (div1, div2) = components((plot1, plot2))
  script, div = components(plot)

  return render_template('alternatives.html', script=script, div=div)


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
            featurestring+='traveltime'
        elif f=='walkability':
            featurestring+='walkability'
        elif f=='roi':
            featurestring+='roi'
        else:
            pass
        if i<len(featurelist)-1:
            featurestring+=','
        
    criteriastring='zprice is not NULL and refshape.refshpindex in ('+ ','.join([str(i) for i in refshpindexlist])+')'

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
  app.run(port=33507)
#  app.run(debug=True,port=5001)
