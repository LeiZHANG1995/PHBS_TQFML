Project description：
This project aims to select efficient factors (such as the supply and demand of bitcoin market, the attractiveness of bitcoin to investors, and the macroeconomic variables) that influence bitcoin price and predict its price changes in the future.

Steps:
1、	Preprocess data
2、	Select efficient factors using SBS algorithms and reduce dimensionality
3、	Use logistic/SVM/KNN or their combination via majority vote to predict the price change of bitcoin
4、	Use k-fold cross-validation to assess model performance and fine-tuning machine learning models via grid search
5、	Diagnose bias and variance problems with learning and validation curves
6、	Evaluate the performance and plot the outcome

Data source:
https://blockchain.info/for data on bitcoin price & supply and demand of bitcoin market
https://www.quandl.com/ for data on the circulation of bitcoin
https://tools.wmflabs.org for data on the amount of bitcoin browsing
Wind for macroeconomic data

