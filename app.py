from flask import Flask, render_template, request, redirect
import csv
import pandas as pd

app = Flask(__name__)

@app.route('/')
def main():
  return redirect('/index')

@app.route('/index')
def index():
  return render_template('index.html')


myData=pd.read_csv('myData.csv')

#predHousePrice_bySAT

if __name__ == '__main__':
  app.run(port=33507)
