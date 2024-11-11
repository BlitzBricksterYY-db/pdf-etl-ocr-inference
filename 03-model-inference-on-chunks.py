# Databricks notebook source
# MAGIC %md
# MAGIC ## 03 - Efficient model inference on chunks
# MAGIC
# MAGIC To perform efficient model inference on each text chunk from the OCRed PDFs, we implement a `pandas_udf` which parallelizes model inference natively on a Spark dataframe.
# MAGIC
# MAGIC We recommend deploying your registered MLFlow model as a [databricks model serving](https://docs.databricks.com/en/machine-learning/model-serving/index.html) endpoint and using this endpoint for inference. This is because these endpoints are highly efficient and auto-scaling. You don't have to optimize compute local to your notebook or cluster to get optimal inference processing.
# MAGIC
# MAGIC In this example we'll use a databricks hosted embedding model to show how we can call this model serving endpoint inside a `pandas_udf` for efficient inference.

# COMMAND ----------

# MAGIC %pip install mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG 'yyang';
# MAGIC USE SCHEMA ner;

# COMMAND ----------

chunks = spark.sql("SELECT * FROM chunks")
display(chunks)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Databricks BGE Embeddings Foundation Model endpoints
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-data-prep-4.png?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC Foundation Models are provided by Databricks, and can be used out-of-the-box.
# MAGIC
# MAGIC Databricks supports several endpoint types to compute embeddings or evaluate a model:
# MAGIC - A **foundation model endpoint**, provided by Databricks (ex: DBRX, Llama3-70B, MPT, BGE). **This is what we'll be using in this demo.**
# MAGIC - An **external endpoint**, acting as a gateway to an external model (ex: Azure OpenAI)
# MAGIC - A **custom**, fined-tuned model hosted on Databricks model service
# MAGIC
# MAGIC Open the [Model Serving Endpoint page](/ml/endpoints) to explore and try the foundation models.
# MAGIC
# MAGIC For this demo, we will use the foundation model `BGE` (embeddings). <br/><br/>
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/databricks-foundation-models.png?raw=true" width="600px" >

# COMMAND ----------

# MAGIC %md
# MAGIC Here is a simple example of inference from a databricks foundation embedding model: `databricks-bge-large-en`

# COMMAND ----------

import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

#Embeddings endpoints convert text into a vector (array of float). Here is an example using BGE:
response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": ["What is Apache Spark?"]})
embeddings = [e['embedding'] for e in response.data]
print(embeddings)

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's wrap that code up into a [`pandas_udf`](https://docs.databricks.com/en/udf/pandas.html) to perform super-efficient model inference natively on a spark dataframe.

# COMMAND ----------

# MAGIC %md
# MAGIC Using Iterator[pd.Series] in a Pandas UDF has several advantages over using pd.Series directly, especially in the context of large-scale data processing in Spark:
# MAGIC
# MAGIC Batch Processing:
# MAGIC
# MAGIC Iterator: Processes data in batches, which can be more efficient for large datasets. It allows the function to handle one batch at a time, reducing memory usage and potentially improving performance.
# MAGIC pd.Series: Processes the entire column at once, which can be less efficient and more memory-intensive for large datasets.
# MAGIC State Initialization:
# MAGIC
# MAGIC Iterator: Allows for initializing state once per batch, which can be useful for operations that require loading models or other resources. This can reduce overhead compared to initializing state for each row.
# MAGIC pd.Series: Does not inherently support state initialization per batch, which can lead to repeated initialization and higher overhead.
# MAGIC Resource Management:
# MAGIC
# MAGIC Iterator: Facilitates better resource management by allowing the use of context managers or try/finally blocks to ensure resources are released after processing each batch.
# MAGIC pd.Series: Managing resources can be more challenging as the entire column is processed at once.

# COMMAND ----------

import mlflow.deployments
import pandas as pd
from typing import Iterator
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, FloatType
# Define a Pandas UDF that returns an array of floats



@pandas_udf(ArrayType(FloatType()))
def call_databricks_model(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
  deploy_client = mlflow.deployments.get_deploy_client("databricks")
  results = []
  for batch in iterator:
    batch_list = batch.to_list()
    sub_batch_size = 150
    sub_batch = [batch_list[i:i + sub_batch_size] for i in range(0, len(batch_list), sub_batch_size)]
    # Process the chunk of 150 items
    for texts in sub_batch:
      response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": texts})
      embeddings = [e['embedding'] for e in response.data]
      results += embeddings
  yield pd.Series(results)


# if the 150 limit didn't exist, this simpler pandas_udf would work.
#
# @pandas_udf(ArrayType(FloatType()))
# def call_databricks_model(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
#   deploy_client = mlflow.deployments.get_deploy_client("databricks")
#   results = []
#   for batch in iterator:
#     response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": batch.to_list()})
#     embeddings = [e['embedding'] for e in response.data]
#     results += embeddings
#   yield pd.Series(results)

# COMMAND ----------

# MAGIC %md
# MAGIC Applying model inference simply on the dataframe by using the `.withColumn` method and the `pandas_udf` declared above.

# COMMAND ----------

inference_df = chunks.withColumn('inference', call_databricks_model('chunk'))
print(inference_df.count())
display(inference_df)

# COMMAND ----------


