from __future__ import division
from flask import Flask, render_template, request, redirect

#import os
#import psycopg2
#import urlparse

#import math
#import numpy as np
#from sklearn.linear_model import LinearRegression
#from sklearn.preprocessing import Imputer

#from bokeh.plotting import *
#from bokeh.embed import components
#from bokeh.models import HoverTool,sources
#from collections import OrderedDict

#from wordcloud import WordCloud
#import matplotlib.pyplot as plt
#import mpld3
#import json

#@app.route('/index', methods=['POST', 'GET'])
@app.route('/index')
def index():
  return render_template('index.html',msg='No such locales found')


# RUN
if __name__ == '__main__':
  app.run(port=33507)
#  app.run(debug=True,port=5001)
