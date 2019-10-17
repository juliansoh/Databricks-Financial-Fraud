# Databricks notebook source
# MAGIC %md #<img width="635" height="132" src="https://www.sao.wa.gov/wp-content/uploads/2018/09/WSAO_LogoLargerIcon-1.png">
# MAGIC #Financial Fraud Audit
# MAGIC October 2019<br>Microsoft Corporation

# COMMAND ----------

# MAGIC %md ##Executive Summary

# COMMAND ----------

# MAGIC %md This is a demonstration of how Databricks can be used as a collaboration and analysis tool for detecting fraud in large datasets.
# MAGIC The dataset used in this demo is the <a href="https://www.kaggle.com/ntnu-testimon/paysim1/download">Kaggle Synthetic Financial Datasets for Fraud Detection</a>
# MAGIC 
# MAGIC This notebook helps the discussion of the following steps:
# MAGIC <ul>
# MAGIC   <li>Modern data access and preparation</li>
# MAGIC   <li>Reviewing, analyzing, and visualizing the data</li>
# MAGIC   <li>Developing Machine Learning Models</li>

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
# MAGIC ### Step 2: Read the data
# MAGIC 
# MAGIC Now that we have specified our file metadata, we can create a DataFrame. Notice that we use an *option* to specify that we want to infer the schema from the file. We can also explicitly set this to a particular schema if we have one already.
# MAGIC 
# MAGIC First, let's create a DataFrame in Python.

# COMMAND ----------

df = spark.read.format(file_type).option("inferSchema", "true").option("header","true").load(file_location)

# COMMAND ----------

# MAGIC %md Examine the detected schema

# COMMAND ----------

df.printSchema()

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

# MAGIC %md For this, we will use PySpark. This is a combination of Python and Spark to address big data. <a href="https://www.edureka.co/blog/pyspark-programming/">Find out more about PySpark</a>.<br>
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

# MAGIC %md ##Internal use only - File Operations

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/FileStore/tables/csv/sao"))

# COMMAND ----------

dbutils.fs.mkdirs("/FileStore/tables/csv/sao")

# COMMAND ----------

# MAGIC %fs wget https://www.sao.wa.gov/wp-content/uploads/2018/09/WSAO_LogoLargerIcon-1.png
# MAGIC dbutils.fs.cp 