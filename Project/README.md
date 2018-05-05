# Final Project Proposal

## Repository name: 
Bitcoin Price Prediction

## Group Member:
* Hang ZHANG        1701213146
* Wei ZHANG         1701213160
* Linsheng ZHUANG   1701212990
* Lei ZHANG         1701213148

## Project description：
This project aims to select efficient factors (such as the supply and demand of bitcoin market, the attractiveness of bitcoin to investors, and the macroeconomic variables) that influence bitcoin price and predict its price changes in the future.

## Goals:
Following the research motivation mentioned above, we have 3 main goals, which are in accordance with our research procedure and clearly state the steps.

1. Time-series analysis: 
For training set, we use ADL model to predict future bitcoin prices roughly on past price by time series analysis.

2. Model modification
Considering that in the real world, ADL model may not fit well, we then use LSTM Neural Network Model to allow for both long-term and short-term repeating patterns and get a more reliable signal about the price rise or fall. 

3. Classification and prediction 
Using the modified predicted price and past value of the class feature (“+1” means that price going up in the following trading day and “-1” means declining), we run a logistic regression on future class feature.

Overall, we hope that our model can help investors learn something about the future trend of bitcoin price (going up/down) from its past trend and then make better investment decisions.

## Data source:
* https://blockchain.info/for data on bitcoin price & supply and demand of bitcoin market
* https://www.quandl.com/ for data on the circulation of bitcoin
* https://tools.wmflabs.org for data on the amount of bitcoin browsing
* Wind for macroeconomic data

