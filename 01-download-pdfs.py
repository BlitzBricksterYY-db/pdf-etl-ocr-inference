# Databricks notebook source
# MAGIC %md
# MAGIC ## 01 - Download PDFs
# MAGIC This notebook creates a PDF dataset to test efficient OCR and inference on PDF documents in the following general steps:
# MAGIC 1. Create a Databricks Volume to store the downloaded files.
# MAGIC 1. Download a [Kaggle dataset](https://www.kaggle.com/datasets/yasirabdaali/arxivorg-ai-research-papers-dataset) which contains links to Arxiv papers on AI.
# MAGIC 1. Loop through the Kaggle dataset to download PDFs using Spark UDFs.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS yyang;
# MAGIC CREATE DATABASE IF NOT EXISTS yyang.ner;
# MAGIC CREATE VOLUME IF NOT EXISTS yyang.ner.data;

# COMMAND ----------

# MAGIC %md
# MAGIC Download the Kaggle dataset (csv file) containing the metadata for 10,000 research papers in the field of artificial intelligence (AI) that were published on arXiv.org. Databricks Volumes act just like a file system, allowing you to run bash commands using the `%sh` magic operator.

# COMMAND ----------

# MAGIC %pip install kaggle --upgrade
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %sh
# MAGIC kaggle datasets download -d yasirabdaali/arxivorg-ai-research-papers-dataset -p /Volumes/yyang/ner/data/ && \
# MAGIC unzip /Volumes/yyang/ner/data/arxivorg-ai-research-papers-dataset.zip -d /Volumes/yyang/ner/data/

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG 'yyang';
# MAGIC USE SCHEMA ner;

# COMMAND ----------

VOLUME = "/Volumes/yyang/ner/data/"

# COMMAND ----------

# MAGIC %md
# MAGIC The dataset has some non-standard csv issues, like newline characters, that affect our ability to read it. This code cleans up the majority of the dataset.

# COMMAND ----------

# clean dataset by removing all \n inside cells
import csv
with open(f"{VOLUME}arxiv_ai.csv", newline="") as input, \
     open(f"{VOLUME}arxiv_ai_cleaned.csv", "w", newline="") as output:
    w = csv.writer(output)
    for record in csv.reader(input):
        w.writerow(tuple(s.replace("\n", " ") for s in record))

# COMMAND ----------

# MAGIC %md
# MAGIC Load the data into a Spark Dataframe.

# COMMAND ----------

df = spark.read.format("csv").option("header", True).load(f"{VOLUME}arxiv_ai_cleaned.csv")
print(df.count())
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Create a function to download the pdfs from the urls in the dataframe, and wrap the function in a Spark UDF so it runs natively on the dataframe.
# MAGIC
# MAGIC Also create a function to store the file sizes. This will help us process the files as efficiently as possible in the future by OCRing the largest files first.

# COMMAND ----------

import os
import sys
import requests
import pandas as pd

from pyspark.sql.functions import col, pandas_udf, udf
from pyspark.sql.types import StringType, ArrayType, LongType
from typing import Iterator, Tuple

if not os.path.exists(f"{VOLUME}pdfs/"):
    os.makedirs(f"{VOLUME}pdfs/")

def download_pdf(url, filename):
  """
  Downloads a PDF file from the given URL and saves it with the specified filename.
  
  Args:
    url (str): The URL of the PDF file to download.
    filename (str): The name of the file to save the PDF as.
      
  Returns:
    bool: True if the PDF file was downloaded successfully, False otherwise.
  """
  if os.path.isfile(filename):
    print(f"Alread exists, skipping: {filename}")
    return True
  try:
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
      # Open a file for writing binary data
      with open(filename, 'wb') as file:
        # Write the PDF content to the file
        file.write(response.content)
      print(f"PDF file downloaded successfully: {filename}")
      return True
    else:
      print(f"Failed to download PDF file. HTTP status code: {response.status_code}")
      return False
  except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
    return False

@udf(returnType=StringType())
def get_pdf_content(url, title, date):
  """
  Creates a filename based off of the paper title and publish date and passes that to the download_pdf function.
  
  Args:
    url (str): The URL of the PDF file to download.
    title (str): The title of the paper.
    date (str): The date the paper was published.
      
  Returns:
    str: The constructed filename for storage back in the dataset.
  """
  date = date[:10]
  filename = "%spdfs/%s_%s.pdf" % (VOLUME, date, title.replace(' ','_'))
  try:
    download_pdf(url, filename)
    result = filename
  except:
    print("Unable to download from {url}, {filename}")
    result = None
  return result

@udf(returnType=LongType())
def get_file_size(filename):
  """
  Creates a filename based off of the paper title and publish date and passes that to the download_pdf function.
  
  Args:
    filename (str): The path/name of the file from which to pull the file size.
      
  Returns:
    int: A integer representing the size of the file in bytes.
  """
  try:
    result = os.path.getsize(filename)
  except:
    result = None
  return result


# COMMAND ----------

# MAGIC %md
# MAGIC Apply the Spark UDFs that we created above natively on the dataset. We also filter to ensure we're only sending properly formed urls to the functions. We also save this dataframe as a delta table for reference in subsequent processing in future notebooks.
# MAGIC
# MAGIC **Note:** We limit this to 100 pdfs for testing purposes.

# COMMAND ----------

from pyspark.sql.functions import col

papers = (df.select(['entry_id','title','pdf_url','published'])
            .limit(100)
            .filter(col("pdf_url").isNotNull())
            .filter("SUBSTRING(pdf_url, 1, 7) = 'http://'")
            .withColumn('filename', get_pdf_content(col('pdf_url'), col('title'), col('published')))
            .withColumn('filesize', get_file_size(col('filename')))
            .write
            .mode('overwrite')
            .saveAsTable('papers')
           )

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM papers;

# COMMAND ----------

# MAGIC %md
# MAGIC The difference between spark parallel vs. Ray (MPI/openMP) parallel
# MAGIC 1. spark prefers data intensive operation
# MAGIC 2. Ray prefers computation or heavy-weight functions intensive operation

# COMMAND ----------

# MAGIC %md
# MAGIC Below is the code I initially thought would be most efficient for OCRing the downloaded PDFs via a [`pandas_udf`](https://docs.databricks.com/en/udf/pandas.html) running on the Spark dataframe. However, the massive efficiency you get from `pandas_udfs` is best for parallelizing data-intensive processing, not for running heavy-weight functions on light data. This is a perfect example of needing a heavy-weight OCR function running on <=10,000 rows.
# MAGIC
# MAGIC I did want to keep this code as a reference for how to create a `pandas_udf` because they are still something powerful to have in your tool belt.
# MAGIC
# MAGIC To better parallelize for efficiency, the next notebook (02-parallel-ocr-with-ray) utilizes [Ray on Databricks](https://docs.databricks.com/en/machine-learning/ray-integration.html) for truly parallelizing the OCR process.

# COMMAND ----------

# from pyspark.sql.functions import explode
# from unstructured.partition.pdf import partition_pdf
# from unstructured.partition.text import partition_text
# from unstructured.chunking.title import chunk_by_title

# @udf(ArrayType(StringType()))
# def parse_pdf(location: pd.Series) -> pd.Series:
#   elements = partition_pdf(location, strategy="ocr_only")
#   chunks = chunk_by_title(elements)
#   return pd.Series([x for x in chunks if len(x) > 50])

# @pandas_udf(ArrayType(StringType()))
# def parse_pdf(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
#   results = []
#   for locations in iterator:
#     for loc in locations:
#       elements = partition_pdf(loc, strategy="ocr_only")
#       # elements = partition_pdf(loc, strategy="hi_res")
#       chunks = chunk_by_title(elements)
#       results.append([str(x) for x in chunks if len(str(x)) > 50])
#   yield pd.Series(results)

# test_df = (papers.limit(2).withColumn('chunk_array', parse_pdf('filename'))
#                  .withColumn('chunks', explode('chunk_array'))
#           )

# display(test_df)

# COMMAND ----------


