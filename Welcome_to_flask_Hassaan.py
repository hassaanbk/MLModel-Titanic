#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 20:18:31 2022

@author: Hassaan
"""

from flask import Flask

app = Flask(__name__)

@app.route("/")

def hello():
    return "Welcome to machine learning model APIs! Hassaan"

if __name__ == '__main__':
    app.run(debug=True, port=12345)