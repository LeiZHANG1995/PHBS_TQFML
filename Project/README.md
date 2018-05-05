# Final Project Proposal

## Repository name: 
Bitcoin Price Prediction

## Group Member:
* Hang ZHANG        1701213146
* Wei ZHANG         1701213160
* Linsheng ZHUANG   1701212990
* Lei ZHANG         1701213148


## Motivation:
Over the last few years, Bitcoin has attracted a lot of investors and researchers due to its growing market capitalization, rapidly increasing price and high price volatility. This project uses efficient factors based on existing research to predict Bitcoin price change by simple LSTM model. 

According to empirical analysis (Ciaian, 2016), Bitcoin market supply-demand fundamentals and Bitcoin’s attractiveness for investors have significant impact on Bitcoin price, while the relevance of macro-financial indicators are statistically insignificant. So we mainly use data for first two dimensions.

## Goals:
We have 3 main goals, which are in accordance with our research procedure and clearly state the steps.

1. Time-series analysis: 
For training set, we use ADL model to predict future bitcoin prices roughly on past price and relevant indicators by time series analysis.

2. Model modification: 
Considering that in the real world, ADL model may not fit well, we then use simple LSTM Neural Network Model to allow for both long-term and short-term repeating patterns and get a more reliable signal about the price rise or fall. 

3. Classification and prediction: 
Using the modified predicted price and past value of the class feature (“+1” means that price going up in the following trading day and “-1” means declining), we run a logistic regression on future class feature.

Overall, we hope that our model can help investors learn something about the future trend of bitcoin price (going up/down) from its past trend and make better investment decisions.

## Data source:
We picked 355 day’s data consisting of 6 variables from May 2nd, 2017 to May 1st, 2018 on a daily basis. The explained variable in our research is the price of bitcoins (“price”). About the features used, we mainly divided the variables into two groups.

First group of features are related to demand and supply of bitcoins. 
* the number of bitcoins circulated in the market (“supply”) measures total supply;
* the total USD value of bitcoin supply in circulation (“capital”) indicates the market size;
* the total USD value of trading volume on major bitcoin exchanges (“trade”) provides trading information.

All the above variables come from https://blockchain.info/for and depict the demand and supply of bitcoins. Also we include “date” as a variable for time series analysis.

Second group of features depicts the attractiveness of bitcoins to investors. We adopted views on Wikipedia (wiki) from https://tools.wmflabs.org/pageviews. We didn’t use Google’s search data because it’s on a weekly basis which does not meet our requirement.
