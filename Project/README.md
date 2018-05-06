# Final Project Proposal

## Repository name: 
Bitcoin Price Prediction

## Group Member:
* Hang ZHANG        1701213146
* Wei ZHANG         1701213160
* Linsheng ZHUANG   1701212990
* Lei ZHANG         1701213148


## Motivation:
Over the last few years, Bitcoin has attracted a lot of investors and researchers due to its growing market capitalization, rapidly increasing price and high price volatility. Understanding Bitcoin price formation and particularly predicting its price change are meaningful for investment decision. This project attempts to employ simple LSTM model to Bitcoin price prediction.

We use efficient factors based on existing research to predict Bitcoin price. According to empirical analysis (Ciaian, 2016), Bitcoin market supply-demand fundamentals and Bitcoin’s attractiveness for investors have significant impact on Bitcoin price, while the relevance of macro-financial indicators are statistically insignificant. So we mainly use data for the first two dimensions.


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
The whole data collection procedure is in [Bitcoin_1_Download_Data.ipynb](https://github.com/LeiZHANG1995/PHBS_TQFML/blob/master/Project/Bitcoin_1_Download_Data.ipynb) file. Please click the link to find more about data collection. In general, we picked 355 day’s data consisting of 6 variables from May 2nd, 2017 to May 1st, 2018 on a daily basis. The explained variable in our research is the price of bitcoins (“price”). About the features used, we mainly divided the variables into two groups.

First group of features are related to demand and supply of bitcoins. 
* the number of bitcoins circulated in the market (“supply”) measures total supply;
* the total USD value of bitcoin supply in circulation (“capital”) indicates the market size;
* the total USD value of trading volume on major bitcoin exchanges (“trade”) provides trading information.

All the above variables come from https://blockchain.info/for and depict the demand and supply of bitcoins. Also we include “date” as a variable for time series analysis.

Second group of features depicts the attractiveness of bitcoins to investors. We adopted views on Wikipedia (wiki) from https://tools.wmflabs.org/pageviews. We didn’t use Google’s search data because it’s on a weekly basis which does not meet our requirement.

## Models
Briefly, we combine the traditional classifiation algorithm (Logistic Regression) and time series regression (ADL model) together. We call our original algorithm **LSTM** model, which is inspired from the LSTM neural network model. If you are interested in the realization of this method, you can find all the details in the [LSTM.py](https://github.com/LeiZHANG1995/PHBS_TQFML/blob/master/Project/LSTM.py) file. The results of data analysis are all displayed in [Bitcoin_2_Data_Analysis.ipynb](https://github.com/LeiZHANG1995/PHBS_TQFML/blob/master/Project/Bitcoin_2_Data_Analysis.ipynb) file, and please click the link to find more details. 

### ADL model
ADL (Autoregressive distributed lag) model is given in the following way:

\[Y_t = \beta (1+L+L^2+...+L^{p-1})Y_{t-1}+\alpha (1+L+L^2+...+L^{p-1})X_{t-1}\]

This model combine both prediction and explanation together, and is the "best" (Professor Jiaxiang Zhu) time series model in econometrics' view. This model is evaluated in two steps:

1) Do first difference on both sides

2) Using OLS to estimate the parameters





