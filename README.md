# Statistical-Model-using-R-for-Bank-Customers

Developed a Classifier to predict whether a customer is going to open a deposit account.

Data: Bank Dataset from UCI Repository https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

Four separate classifiers (Decision Tree, Naïve Bayes, SVM, Neural Network) have been built and compared their performance in terms of Accuracy, Sensitivity, Specificity and Precision using 10 fold cross validation.

Developed a weighted F measure to reflect selection. Calculated the weighted F models for the 4 models developed in part 1 and recommended the best model based on weighted F Measure.

Performed Dimensionality reduction on the numeric metrics using PCA and analyzed the effect of PCA on classifier performance.
Comparative performance of different algorithms with/without PCA

## Data Preprocesing:
Firstly, data is pre-processed to remove redundancy and to improve the 
performance of the model:
1.	Eliminated attributes which has Correlation >0.9.
2.	Checked for NullVariance and corresponding attributes are removed.
3.	Factors of length 1 are removed.
4.	There are 18 attributes that are finally left out of 21 after pre-processing.
5.	40% are data are stratified sampled and are used as training dataset and 
rest 60% data is used as test dataset. This proportion of dataset is chosen
after testing various proportions of dataset(like 75%:25%),(60%:40%), out 
of which (40-training): (60-testing) executed in efficient timeframe during 
training.

## With 10-fold cross validation:

Model           Accuracy	Sensitivity	Specificity	Precision	  F1	    F0.5
Neural Networks	0.8873	  0.9999	    0.0003	    0.8873	  0.9403	0.9077
SVM	0.9045	    0.9797	  0.3125	    0.9182	    0.9479	  0.9479  0.9298
Decision Tree	  0.9098	  0.9746	    0.3997	    0.9275	  0.9504	0.9365
Naïve Bayes	    0.8988	  0.9711	    0.254	      0.8774	  0.9401	0.8946

## Recall vs Precision:

“Precision” is considered to be of higher importance in this case, since precision is average probability of relevant retrieval which makes sure that there no false positive results, while recall is the average probability of the complete retrieval that makes sure there are no false Negative result. For example, I cannot have a lot of applications falsely tested for opening a term deposit, if I know that all of the ones who are willing to open a term deposit are tested. In this case, all the relevant hits are identified so that there are no false positive results. Precision can test and make sure that opening a term deposit for an unsanctioned or otherwise undesirable party can incur large fine or even endanger banking license and thus will prevent them.  Also, Staffs required to review high volumes of false positive results may inadvertently overlook a true crucial positive hit.
Since, Precision is preferred for this case, Beta =0.5(0<beta<1) 
So Weighted-F becomes:
F0.5=(1.25*recall *precision)/(0.25*precision)+recall

•	Based on F1 value (which considers recall and precision to be of equal importance), the leader by marginal difference beating SVM is “Decision Tree”.

•	Also, Weighted F-value (F0.5 value) is more for Decision tree compared to any other models. 

Hence, recommended model with 10-fold cross validation would be 
“Decision tree”.

## Dimensionality Reduction With PCA:

Model             Accuracy	Sensitivity	Specificity	Precision	F1	  F0.5
Neural Networks	  0.8873	  0.9999	    0.0003	    0.8873	0.9474	0.907744449
SVM   	          0.899	    0.9839	    0.2298	    0.9096	0.9453	0.923548502
Decision Tree   	0.9027	  0.9693	    0.3785	    0.9247	0.9464	0.933288605
Naïve Bayes	      0.888	    0.9611	    0.194	      0.8564	0.9371	0.87547443

## Comparing Without and With PCA:
Model                      Accuracy	Sensitivity	Specificity	Precision	F1	  F0.5
Neural Networks	            0.8873	0.9999	      0.0003	  0.8873	0.9403	0.9077
Neural Networks(With PCA)	  0.8873	0.9999	      0.0003	  0.8873	0.9474	0.9077
SVM	                        0.9045	0.9797	      0.3125	  0.9182	0.9479	0.9299
SVM(With PCA)	              0.899	  0.9839	      0.2298	  0.9096	0.9453	0.9235
Decision Tree	              0.9098	0.9746	      0.3997	  0.9275	0.9504	0.9366
Decision Tree(With PCA)	    0.9027	0.9693	      0.3785	  0.9247	0.9464	0.9333
Naïve Bayes	                0.8988	0.9711	      0.254	    0.8774	0.9401	0.8947
Naïve Bayes(With PCA)	      0.888	  0.9611	      0.194	    0.8564	0.9371	0.8755

## Inference:
Neural networks have no impact with and without PCA, while the other three model’s accuracy, precision, specificity, F1 and Measured F values have been marginally reduced with PCA and takes more time for execution.
Even with PCA, Decision Tree has higher F1 and F2 values with more accuracy and precision compared to other models.
