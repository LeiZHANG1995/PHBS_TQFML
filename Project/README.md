# Final Project: Bitcoin Price Prediction

## Group Member
* Linsheng ZHUANG   1701212990
* Hang ZHANG        1701213146
* Lei ZHANG         1701213148
* Wei ZHANG         1701213160

## Motivation
Over the last few years, Bitcoin has attracted a lot of investors and researchers due to its growing market capitalization, rapidly increasing price and high price volatility. Understanding Bitcoin price formation and particularly predicting its price change are meaningful for investment decision. This project attempts to employ simple LSTM model to Bitcoin price prediction.

We use efficient factors based on existing research to predict Bitcoin price. According to empirical analysis (Ciaian, 2016), Bitcoin market supply-demand fundamentals and Bitcoin’s attractiveness for investors have significant impact on Bitcoin price, while the relevance of macro-financial indicators are statistically insignificant. So we mainly use data for the first two dimensions.


## Goals
Ultimately, we want to predict the price fluctuation of bitcoin. We want to do forecast about the 'up' and 'down' of price tendency in the future. We will do this in three steps:

**1. Time-series Model: predict the future price (hidden layer).**  For training set, we use ADL model to predict future bitcoin prices roughly on past price and relevant indicators by time series analysis.

**2. Model Modification by Classification: up and down (output).**  Considering that in the real world, ADL model may not fit well, we then use simple LSTM Neural Network Model to allow for both long-term and short-term repeating patterns and get a more reliable signal about the price rise or fall (“+1” means that price going up in the following trading day and “-1” means declining).

**3. Prediction.**  Using the modified predicted price and past value of the class feature , we run our model on the test data set.

Overall, we hope that our model can help investors learn something about the future trend of bitcoin price (going up/down) from its past trend and make better investment decisions.

## Data source
The whole data collection procedure is in [Bitcoin_1_Download_Data.ipynb](https://github.com/LeiZHANG1995/PHBS_TQFML/blob/master/Project/Bitcoin_1_Download_Data.ipynb) file. Please click the link to find more about data collection. In general, we picked 355 day’s data consisting of 6 variables from May 2nd, 2017 to May 1st, 2018 on a daily basis. The explained variable in our research is the price of bitcoins (“price”). About the features used, we mainly divided the variables into two groups.

First group of features are related to demand and supply of bitcoins. 
* the number of bitcoins circulated in the market (“supply”) measures total supply;
* the total USD value of bitcoin supply in circulation (“capital”) indicates the market size;
* the total USD value of trading volume on major bitcoin exchanges (“trade”) provides trading information.

All the above variables come from https://blockchain.info/for and depict the demand and supply of bitcoins. Also we include “date” as a variable for time series analysis.

Second group of features depicts the attractiveness of bitcoins to investors. We adopted views on Wikipedia (wiki) from https://tools.wmflabs.org/pageviews. We didn’t use Google’s search data because it’s on a weekly basis which does not meet our requirement.

## Model
Briefly, we combine the traditional classifiation algorithm (Logistic Regression) and time series regression (ADL model) together. We call our original algorithm **LSTM** model, which is inspired from the [LSTM Neural Network Model](http://www.jakob-aungiers.com/articles/a/LSTM-Neural-Network-for-Time-Series-Prediction). If you are interested in the realization of this method, you can find all the details in the [LSTM.py](https://github.com/LeiZHANG1995/PHBS_TQFML/blob/master/Project/LSTM.py) file. The results of data analysis are all displayed in [Bitcoin_2_Data_Analysis.ipynb](https://github.com/LeiZHANG1995/PHBS_TQFML/blob/master/Project/Bitcoin_2_Data_Analysis.ipynb) file, and please click the link to find more details. 

### ADL model
ADL (Autoregressive distributed lag) model is given in the following form:

$$ P_t = \beta (1+L+L^2+...+L^{k-1})P_{t-1}+\alpha (1+L+L^2+...+L^{k-1})X_{t-1} $$

This model combines both prediction and explanation together, and is the "best" (Professor Jiaxiang Zhu) time series model in econometrics' view. This model is evaluated in two steps:

1) Do first order difference on both sides of the regression equation

2) Using OLS to estimate the parameters: using QR decomposition to improve accuracy

After estimation, we can use the estimated model to do prediction. If the `lag` order is _k_, given _P_ from 1 to _k_ period, we are able to predict the _P_ of _k_+1 preiod using this ADL model. 

### LSTM model
From previous step, we can get a set of predicted _P_. Compare those predicted _P_ with the real price, and we can draw some conclusion about the future price fluctuation. But this prediction is not as reliable as we think. 

One we to improve the reliability of this "up and down" judgement is to take the predicted price as the hidden layer, and further do some classifcation on our hidden layer. The final output would combine both the predicted _y_ and other information that we think is of value. This modification might improve the prediction accuracy. 

As for the added information that we consider in the second layer (classification), we add the price fluctuation of the previous period. This makes our model a little bit similar to the _Long and Short Term Memory_ neural network model. 

### Fit the data window recursively
This is not enough. The time span of our data is one year, and the data pattern may differ in different time periods. For example, the price tendency goes up in the first 3 months, and then goes down in the next 4 months. Both ADL regression and Logistic classification are linear model, therefore, the trained model of the pooled training data does not seem reliable.

To fix this problem, Our group choose a recursive way of estimation and prediction. We do our LSTM algorithm on moving data windows. The window length can be 40 days or 50 days. We first select the window from 1-40 period and do LSTM estimation, and then we move the data window to 2-41 period, and repeat this procedure. 

This recursive window way can be implement on both the **training set** and **test set**. It can improve the performance of training accuracy, but it can cause overfitting problems if we selece a short window length. 

## Conclusion
The accuracy of ADL model ranges from 0.52 to 0.65 (training set) and 0.46 to 0.55 (test set). Considering the model performance, we set the lag to be 2, indicating that the previous-two-days prices have significant influence over the current price. In this case pooled ADL is stationary, and does not face overfitting problem. But the accuracy is just a little bit higher than 0.5. On the other hand, the accuracy of prediction relies heavily on the training sample size and the accuracy of training set and test set do not move along the same direction.

After the use of LSTM model, we can see that the consideration of memory mechanism improved the prediction accuracy a lot, which can reach 0.65 in training set and 0.60 in test set. However, LSTM is very sensitive to the input parameters, and the model can be overfitting in many cases, but it predicts well under some optimal parameters.

Overall speaking, the prediction result of LSTM model can provide reasonable advice for the investment decisions. If we get +1 from the historical data, it means that the possibility that bitcoin price will go up in the next day is around 60%, vice versa. However, the peak of our prediction accuracy never exceeds 0.7, which means that real world is much more complicated than our model. There is a limitation to econometric models due to complex environments and different subjective opinions.
