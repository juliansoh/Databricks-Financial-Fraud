# Databricks notebook source
# MAGIC %md #<a href="http://www.sao.wa.gov"><img width="635" height="132" src="https://www.sao.wa.gov/wp-content/uploads/2018/09/WSAO_LogoLargerIcon-1.png"></a>
# MAGIC #Financial Fraud Audit
# MAGIC ***(Exercise. Not an actual Fraud Audit)***<br><br>
# MAGIC <a href="https://www.linkedin.com/in/julian-soh-307ab24/">Julian Soh</a><br>
# MAGIC October 2019<br>Microsoft Corporation

# COMMAND ----------

# MAGIC %md ##Executive Summary

# COMMAND ----------

# MAGIC %md This is a demonstration of how Databricks can be used as a collaboration and analysis tool for detecting fraud in large datasets.
# MAGIC The dataset used in this demo is the [Kaggle Synthetic Financial Datasets for Fraud Detection](https://www.kaggle.com/ntnu-testimon/paysim1/download)
# MAGIC 
# MAGIC This notebook helps the discussion of the following steps:
# MAGIC <ul>
# MAGIC   <li>Modern data access and preparation</li>
# MAGIC   <li>Reviewing, analyzing, and visualizing the data</li>
# MAGIC   <li>Developing Machine Learning Models</li>
# MAGIC   
# MAGIC Goals:
# MAGIC <ul>
# MAGIC   <li>Showcase the power of Big Data in Microsoft Azure</li>
# MAGIC   <li>Demonstrate Intelligent Cloud Computing capabilities - Machine Learning and Advanced Analytics</li>
# MAGIC   <li>Introduce Azure Databricks as a web-based Unified Analytics Platform that unites people (collaboration), processes and infrastructure (cluster/platform)</li>

# COMMAND ----------

# MAGIC %md ## Data Access & Preparation

# COMMAND ----------

# MAGIC %md ###Step 1: Define Blob access and location

# COMMAND ----------

storage_account_name = "saadflabblob"
storage_account_access_key = "zzE+5F19+MuCjM8Pje3XwFboffBKdO/7vVEVbNLf/071sCnrx7fgKGgXP4IQ22zRL9GQebKwJ/pbq4i29uKAwQ=="

# COMMAND ----------

file_location = "wasbs://raw@saadflabblob.blob.core.windows.net/"
file_type = "csv"

# COMMAND ----------

spark.conf.set(
  "fs.azure.account.key."+storage_account_name+".blob.core.windows.net",
  storage_account_access_key)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 2: Read and prep the data
# MAGIC 
# MAGIC Now that we have specified our file metadata, we can create a DataFrame. Notice that we use an *option* to specify that we want to infer the schema from the file. We can also explicitly set this to a particular schema if we have one already.
# MAGIC 
# MAGIC First, let's create a DataFrame in Python.

# COMMAND ----------

df = spark.read.format(file_type).option("inferSchema", "true").option("header","true").load(file_location)

# COMMAND ----------

# Display the inferred schema to see if it is correct
df.printSchema()

# COMMAND ----------

#Changing the schema type (if necessary)

df = df.select(
  df.step.cast("integer"),
  df.type.cast("string"),
  df.amount.cast("double"),
  df.nameOrig.cast("string"),
  df.oldbalanceOrg.cast("double"),
  df.newbalanceOrig.cast("double"),
  df.nameDest.cast("string"),
  df.oldbalanceDest.cast("double"),
  df.newbalanceDest.cast("double"),
  df.isFraud.cast("integer"),
  df.isFlaggedFraud.cast("integer")
)

# COMMAND ----------

#Recheck to see if schema has be changed successfully

df.printSchema()

# COMMAND ----------

#How many rows do we have in this DataFrame

df.count()

# COMMAND ----------

# MAGIC %md ## Taxonomy
# MAGIC step - maps a unit of time in the real world. In this case 1 step is 1 hour of time. Total steps 744 (30 days simulation).
# MAGIC 
# MAGIC type - CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.
# MAGIC 
# MAGIC amount - amount of the transaction in local currency.
# MAGIC 
# MAGIC nameOrig - customer who started the transaction
# MAGIC 
# MAGIC oldbalanceOrg - initial balance before the transaction
# MAGIC 
# MAGIC newbalanceOrig - new balance after the transaction
# MAGIC 
# MAGIC nameDest - customer who is the recipient of the transaction
# MAGIC 
# MAGIC oldbalanceDest - initial balance recipient before the transaction. Note that there is not information for customers that start with M (Merchants).
# MAGIC 
# MAGIC newbalanceDest - new balance recipient after the transaction. Note that there is not information for customers that start with M (Merchants).
# MAGIC 
# MAGIC isFraud - This is the transactions made by the fraudulent agents inside the simulation. In this specific dataset the fraudulent behavior of the agents aims to profit by taking control or customers accounts and try to empty the funds by transferring to another account and then cashing out of the system.
# MAGIC 
# MAGIC isFlaggedFraud - The business model aims to control massive transfers from one account to another and flags illegal attempts. An illegal attempt in this dataset is an attempt to transfer more than 200.000 in a single transaction.

# COMMAND ----------

# MAGIC %md #Start of Audit

# COMMAND ----------

# MAGIC %md ##Review the data

# COMMAND ----------

# Review the new table (including the origination and destiation differences)
display(df)

# COMMAND ----------

# Let's add some additional columns for our analysis work

# Calculate the differences between originating and destination balances
df = df.withColumn("orgDiff", df.newbalanceOrig - df.oldbalanceOrg).withColumn("destDiff", df.newbalanceDest - df.oldbalanceDest)

# Create temporary view
df.createOrReplaceTempView("financials")

# COMMAND ----------

# Review the new table (including the origination and destiation differences)
display(df)

# COMMAND ----------

# MAGIC %md ##Types of Transactions

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Organize by Type
# MAGIC Select type, count(1)
# MAGIC From financials
# MAGIC group by type

# COMMAND ----------

# MAGIC %md ##How much money in each type?

# COMMAND ----------

# MAGIC %sql
# MAGIC select type, sum(amount) 
# MAGIC from financials 
# MAGIC group by type

# COMMAND ----------

# MAGIC %md ##Tests for fraud based on known cases
# MAGIC 
# MAGIC The following "where" clause are a set of rules to identify known fraud-based cases using SQL; i.e. rules-based model. SAO staff is familiar with SQL.
# MAGIC <ul>
# MAGIC   <li>Financial fraud analytics often starts with clauses like the "where" clause below</li>
# MAGIC   <li>Note that in reality, rules are often much larger and more complicated. But this is where/how you can perform your current known tests</li>
# MAGIC </ul>

# COMMAND ----------

# MAGIC %md For this, we will use PySpark. This is a combination of Python and Spark to address big data. [Find out more about PySpark](https://www.edureka.co/blog/pyspark-programming/)<br>
# MAGIC  <img src="https://d1jnx9ba8s6j9r.cloudfront.net/blog/wp-content/uploads/2018/07/PySpark.png" width=548 height=74>

# COMMAND ----------

from pyspark.sql import functions as F

# Rules to Identify Known Fraud-based
df = df.withColumn("label", 
                   F.when(
                     (
                       (df.oldbalanceOrg <= 56900) & (df.type == "TRANSFER") & (df.newbalanceDest <= 105)) | 
                       (
                         (df.oldbalanceOrg > 56900) & (df.newbalanceOrig <= 12)) | 
                           (
                             (df.oldbalanceOrg > 56900) & (df.newbalanceOrig > 12) & (df.amount > 1160000)
                           ), 1
                   ).otherwise(0))

# Calculate proportions
fraud_cases = df.filter(df.label == 1).count()
total_cases = df.count()
fraud_pct = 1.*fraud_cases/total_cases

# Provide quick statistics
print("Based on these rules, we have flagged %s (%s) fraud cases out of a total of %s cases." % (fraud_cases, fraud_pct, total_cases))

# Create temporary view to review data
df.createOrReplaceTempView("financials_labeled")

# COMMAND ----------

# MAGIC %md ##Quantify the possible fraud cases
# MAGIC Based on the known tests/rules, we found 4% of the transactions as potentially fraudulant, but this 4% in number of transactions represents 11% of the total transactions in dollars (see below).

# COMMAND ----------

# MAGIC %sql
# MAGIC select label, count(1) as `Transactions`, 
# MAGIC         sum(amount) as `Total Amount` 
# MAGIC from financials_labeled group by label

# COMMAND ----------

# MAGIC %md ##Top Origination / Destination difference pairs (>$1M TotalDesDiff)
# MAGIC Each bar represents a pair of entities performing a transaction

# COMMAND ----------

# MAGIC %sql
# MAGIC -- where sum(destDiff) >= 10000000.00
# MAGIC select nameOrig, nameDest, label, TotalOrgDiff, TotalDestDiff
# MAGIC   from (
# MAGIC      select nameOrig, nameDest, label, sum(OrgDiff) as TotalOrgDiff, sum(destDiff) as TotalDestDiff 
# MAGIC        from financials_labeled 
# MAGIC       group by nameOrig, nameDest, label 
# MAGIC      ) a
# MAGIC  where TotalDestDiff >= 1000000
# MAGIC  limit 100

# COMMAND ----------

# MAGIC %md ##What is the most common type of transactions that are associated to fraud?
# MAGIC Reviewing the rules-based tests for fraud, it appears that most faudulant transactions are in the category of "Transfer" and "Cash_Out"

# COMMAND ----------

# MAGIC %sql
# MAGIC select type, label, count(1) as `Transactions` from financials_labeled group by type, label

# COMMAND ----------

# MAGIC %md ##Using Machine Learning (ML) models

# COMMAND ----------

# MAGIC %md ###Building a ML Fraud model

# COMMAND ----------

# Initially split our dataset between training and test datasets
#We will take 80% of the dataset for training purposes and set aside 20% of the dataset for validation
(train, test) = df.randomSplit([0.8, 0.2], seed=12345)

# Cache the training and test datasets
train.cache()
test.cache()

# Print out dataset counts
print("Total rows: %s, Training rows: %s, Test rows: %s" % (df.count(), train.count(), test.count()))

# COMMAND ----------

# MAGIC %md ###An ML Pipeline
# MAGIC To create an ML model, this entails repeating steps (e.g. StringIndexer, VenctorAssmbler, etc). An ML pipeline allows you to reuse these steps with new/additional data for retraining purposes. The key is thaat the more data you have, the more accurate your ML model will be.

# COMMAND ----------

#Create the ML Pipeline

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier

# Encodes a string column of labels to a column of label indices
indexer = StringIndexer(inputCol = "type", outputCol = "typeIndexed")

# VectorAssembler is a transformer that combines a given list of columns into a single vector column
va = VectorAssembler(inputCols = ["typeIndexed", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "orgDiff", "destDiff"], outputCol = "features")

# Using the DecisionTree classifier model
dt = DecisionTreeClassifier(labelCol = "label", featuresCol = "features", seed = 54321, maxDepth = 5)

# Create our pipeline stages
pipeline = Pipeline(stages=[indexer, va, dt])

# COMMAND ----------

# View the Decision Tree model (prior to CrossValidator)
dt_model = pipeline.fit(train)
display(dt_model.stages[-1])

# COMMAND ----------

# MAGIC %md ###Use BinaryClassificationEvaluator
# MAGIC Determine the accuracy of the model by reviewing the "areaUnderPR" and "areaUnderROC"

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Use BinaryClassificationEvaluator to evaluate our model
evaluatorPR = BinaryClassificationEvaluator(labelCol = "label", rawPredictionCol = "prediction", metricName = "areaUnderPR")
evaluatorAUC = BinaryClassificationEvaluator(labelCol = "label", rawPredictionCol = "prediction", metricName = "areaUnderROC")

# COMMAND ----------

# MAGIC %md ###Setup CrossValidation
# MAGIC To try out different parameters to potentially improve our model, we will use CrossValidator in conjunction with the ParamGridBuilder to automate trying out different parameters.
# MAGIC Note, we are using evaluatorPR as our evaluator as the Precision-Recall curve is often better for an unbalanced distribution.

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Build the grid of different parameters
paramGrid = ParamGridBuilder() \
    .addGrid(dt.maxDepth, [5, 10, 15]) \
    .addGrid(dt.maxBins, [10, 20, 30]) \
    .build()

# Build out the cross validation
crossval = CrossValidator(estimator = dt,
                          estimatorParamMaps = paramGrid,
                          evaluator = evaluatorPR,
                          numFolds = 3)  

pipelineCV = Pipeline(stages=[indexer, va, crossval])

# Train the model using the pipeline, parameter grid, and preceding BinaryClassificationEvaluator
cvModel_u = pipelineCV.fit(train)

# COMMAND ----------

# MAGIC %md ###Review Results
# MAGIC Review the areaUnderPR (area under Precision Recall curve) and areaUnderROC (area under Receiver operating characteristic) or AUC (area under curve) metrics

# COMMAND ----------

# Build the best model (training and test datasets)
train_pred = cvModel_u.transform(train)
test_pred = cvModel_u.transform(test)

# Evaluate the model on training datasets
pr_train = evaluatorPR.evaluate(train_pred)
auc_train = evaluatorAUC.evaluate(train_pred)

# Evaluate the model on test datasets
pr_test = evaluatorPR.evaluate(test_pred)
auc_test = evaluatorAUC.evaluate(test_pred)

# Print out the PR and AUC values
print("PR train:", pr_train)
print("AUC train:", auc_train)
print("PR test:", pr_test)
print("AUC test:", auc_test)

# COMMAND ----------

# MAGIC %md ###Confusion Matrix Code-base
# MAGIC Subsequent cells will be using the following code to plot the confusion matrix.

# COMMAND ----------

# Create confusion matrix template
from pyspark.sql.functions import lit, expr, col, column

# Confusion matrix template
cmt = spark.createDataFrame([(1, 0), (0, 0), (1, 1), (0, 1)], ["label", "prediction"])
cmt.createOrReplaceTempView("cmt")

# COMMAND ----------

# Source code for plotting confusion matrix is based on `plot_confusion_matrix` 
# via https://runawayhorse001.github.io/LearningApacheSpark/classification.html#decision-tree-classification
import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm, title):
  # Clear Plot
  plt.gcf().clear()

  # Configure figure
  fig = plt.figure(1)
  
  # Configure plot
  classes = ['Fraud', 'No Fraud']
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  # Normalize and establish threshold
  normalize=False
  fmt = 'd'
  thresh = cm.max() / 2.

  # Iterate through the confusion matrix cells
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  # Final plot configurations
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label') 
  
  # Display images
  image = fig
  
  # Show plot
  #fig = plt.show()
  
  # Save plot
  fig.savefig("confusion-matrix.png")

  # Display Plot
  display(image)
  
  # Close Plot
  plt.close(fig)

# COMMAND ----------

# Create temporary view for test predictions
test_pred.createOrReplaceTempView("test_pred")

# Create test predictions confusion matrix
test_pred_cmdf = spark.sql("select a.label, a.prediction, coalesce(b.count, 0) as count from cmt a left outer join (select label, prediction, count(1) as count from test_pred group by label, prediction) b on b.label = a.label and b.prediction = a.prediction order by a.label desc, a.prediction desc")

# View confusion matrix
display(test_pred_cmdf)

# COMMAND ----------

# MAGIC %md ### View Confusion Matrix
# MAGIC Using matplotlib and pandas to visualize our confusion matrix

# COMMAND ----------

# Convert to pandas
cm_pdf = test_pred_cmdf.toPandas()

# Create 1d numpy array of confusion matrix values
cm_1d = cm_pdf.iloc[:, 2]

# Create 2d numpy array of confusion matrix values
cm = np.reshape(cm_1d, (-1, 2))

# Print out the 2d array
print(cm)


# COMMAND ----------

# Plot confusion matrix  
plot_confusion_matrix(cm, "Confusion Matrix (Unbalanced Test)")

# COMMAND ----------

# Log MLflow
with mlflow.start_run(experiment_id = mlflow_experiment_id) as run:
  # Log Parameters and metrics
  mlflow.log_param("balanced", "no")
  mlflow.log_metric("PR train", pr_train)
  mlflow.log_metric("AUC train", auc_train)
  mlflow.log_metric("PR test", pr_test)
  mlflow.log_metric("AUC test", auc_test)
  
  # Log model
  mlflow.spark.log_model(dt_model, "model")
  
  # Log Confusion matrix
  mlflow.log_artifact("confusion-matrix.png")

# COMMAND ----------

# MAGIC %md ### Model with Balanced classes
# MAGIC Let's see if we can improve our decision tree model but balancing the Fraud vs. No Fraud cases. We will tune the model using the metrics areaUnderROC or (AUC)

# COMMAND ----------

# Reset the DataFrames for no fraud (`dfn`) and fraud (`dfy`)
dfn = train.filter(train.label == 0)
dfy = train.filter(train.label == 1)

# Calculate summary metrics
N = train.count()
y = dfy.count()
p = y/N

# Create a more balanced training dataset
train_b = dfn.sample(False, p, seed = 92285).union(dfy)

# Print out metrics
print("Total count: %s, Fraud cases count: %s, Proportion of fraud cases: %s" % (N, y, p))
print("Balanced training dataset count: %s" % train_b.count())


# COMMAND ----------

# Display our more balanced training dataset
display(train_b.groupBy("label").count())

# COMMAND ----------

# MAGIC %md ### Update the ML Pipeline
# MAGIC Because we had created the ML pipeline stages in the previous cells, we can re-use them to execute it against our balanced dataset.
# MAGIC Note, we are using evaluatorAUC as our evaluator as this is often better for a balanced distribution.

# COMMAND ----------

# Re-run the same ML pipeline (including parameters grid)
crossval_b = CrossValidator(estimator = dt,
                          estimatorParamMaps = paramGrid,
                          evaluator = evaluatorAUC,
                          numFolds = 3)  

pipelineCV_b = Pipeline(stages=[indexer, va, crossval_b])

# Train the model using the pipeline, parameter grid, and BinaryClassificationEvaluator using the `train_b` dataset
cvModel_b = pipelineCV_b.fit(train_b)

# COMMAND ----------

# Build the best model (balanced training and full test datasets)
train_pred_b = cvModel_b.transform(train_b)
test_pred_b = cvModel_b.transform(test)

# Evaluate the model on the balanced training datasets
pr_train_b = evaluatorPR.evaluate(train_pred_b)
auc_train_b = evaluatorAUC.evaluate(train_pred_b)

# Evaluate the model on full test datasets
pr_test_b = evaluatorPR.evaluate(test_pred_b)
auc_test_b = evaluatorAUC.evaluate(test_pred_b)

# Print out the PR and AUC values
print("PR train:", pr_train_b)
print("AUC train:", auc_train_b)
print("PR test:", pr_test_b)
print("AUC test:", auc_test_b)

# COMMAND ----------

# Create temporary view for test predictions
test_pred_b.createOrReplaceTempView("test_pred_b")

# Create test predictions confusion matrix
test_pred_b_cmdf = spark.sql("select a.label, a.prediction, coalesce(b.count, 0) as count from cmt a left outer join (select label, prediction, count(1) as count from test_pred_b group by label, prediction) b on b.label = a.label and b.prediction = a.prediction order by a.label desc, a.prediction desc")

# View confusion matrix
display(test_pred_b_cmdf)

# COMMAND ----------

# Convert to pandas
cm_b_pdf = test_pred_b_cmdf.toPandas()

# Create 1d numpy array of confusion matrix values
cm_b_1d = cm_b_pdf.iloc[:, 2]

# Create 2d numpy array of confusion matrix vlaues
cm_b = np.reshape(cm_b_1d, (-1, 2))

# Print out the 2d array
print(cm_b)


# COMMAND ----------

# MAGIC %md ###View the Decision Tree Models
# MAGIC Visually compare the differences between the unbalanced and balanced decision tree models (basd on the train and train_b datasets respectively).

# COMMAND ----------

# Extract Feature Importance
#  Attribution: Feature Selection Using Feature Importance Score - Creating a PySpark Estimator
#               https://www.timlrx.com/2018/06/19/feature-selection-using-feature-importance-score-creating-a-pyspark-estimator/
import pandas as pd

def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))

# COMMAND ----------

# View the Unbalanced Decision Tree model (prior to CrossValidator)
dt_model = pipeline.fit(train)
display(dt_model.stages[-1])

# COMMAND ----------

# Extract Feature Importance for the original unbalanced dt_model
ExtractFeatureImp(dt_model.stages[-1].featureImportances, train_pred, "features").head(10)

# COMMAND ----------

# View the Balanced Decision Tree model (prior to CrossValidator)
dt_model_b = pipeline.fit(train_b)
display(dt_model_b.stages[-1])

# COMMAND ----------

# Extract Feature Importance for the nbalanced dt_model
ExtractFeatureImp(dt_model_b.stages[-1].featureImportances, train_pred_b, "features").head(10)

# COMMAND ----------

# MAGIC %md ###Comparing Confusion Matrices
# MAGIC Below we will compare the unbalanced and balanced decision tree ML model confusion matrices.

# COMMAND ----------

# Plot confusion matrix  
plot_confusion_matrix(cm, "Confusion Matrix (Unbalanced Test)")

# COMMAND ----------

# Plot confusion matrix  
plot_confusion_matrix(cm_b, "Confusion Matrix (Balanced Test)")

# COMMAND ----------

# Log MLflow
with mlflow.start_run(experiment_id = mlflow_experiment_id) as run:
  # Log Parameters and metrics
  mlflow.log_param("balanced", "yes")
  mlflow.log_metric("PR train", pr_train_b)
  mlflow.log_metric("AUC train", auc_train_b)
  mlflow.log_metric("PR test", pr_test_b)
  mlflow.log_metric("AUC test", auc_test_b)
    
  # Log model
  mlflow.spark.log_model(dt_model_b, "model")
  
  # Log Confusion matrix
  mlflow.log_artifact("confusion-matrix.png")%md ##Internal use only - File Operations

# COMMAND ----------



# COMMAND ----------

#display(dbutils.fs.ls("dbfs:/FileStore/tables/csv/sao"))

# COMMAND ----------

#dbutils.fs.mkdirs("/FileStore/tables/csv/sao")

# COMMAND ----------

#%fs wget https://www.sao.wa.gov/wp-content/uploads/2018/09/WSAO_LogoLargerIcon-1.png
#dbutils.fs.cp 