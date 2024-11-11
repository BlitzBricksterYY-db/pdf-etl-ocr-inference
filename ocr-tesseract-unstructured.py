# Databricks notebook source
# MAGIC %md
# MAGIC ## Offline OCR on Databricks
# MAGIC This notebook installs `tesseract` and other necessary libraries on a single node cluster and utilizes the `unstructured` library to perfome OCR and chunking without hitting the `unstructured` API.
# MAGIC
# MAGIC This performs single-threaded OCR. If you want to do parallel OCR using Ray, refer to [this notebook](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#notebook/2617645319412845/command/2617645319499319).
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ##### init scripts 
# MAGIC We utilize the [`unstructured`](https://unstructured.io/) library to perform OCR and chunking of pdf documents. Unstructured utilizes [tesseract](https://github.com/tesseract-ocr/tesseract) for OCR, which requires tesseract to be installed on every node of the cluster. In order to do that, we could take advantage of databricks [init scripts](https://docs.databricks.com/en/init-scripts/index.html). Init scripts are run at cluster startup and are the best way to install os-level libraries before spark is started on each cluster.
# MAGIC
# MAGIC In this case, it is a very simple, two-line file:
# MAGIC ```bash
# MAGIC apt update
# MAGIC apt-get install -y poppler-utils libmagic-dev tesseract-ocr
# MAGIC ```
# MAGIC
# MAGIC Otherwise, if you use a single-node cluster, you can perform the same installations by running the next cell with the `%sh` magic command.

# COMMAND ----------

# MAGIC %sh
# MAGIC apt update
# MAGIC apt install -y poppler-utils libmagic-dev tesseract-ocr
# MAGIC # if you're working on a single node cluster, you can just run this cell in the notebook instead of using an init script

# COMMAND ----------

# MAGIC %pip install databricks-sdk databricks-vectorsearch unstructured[pdf] --upgrade
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG 'justinm';
# MAGIC CREATE SCHEMA IF NOT EXISTS unstructured;
# MAGIC USE SCHEMA unstructured;

# COMMAND ----------

catalog = "justinm"
db = "unstructured"
volume = "/Volumes/justinm/unstructured/pdfs/state_docs"

# COMMAND ----------

# MAGIC %md
# MAGIC Get the file names of all of PDFs stored in the Volume

# COMMAND ----------

import glob, os

pdfs = []
for file in glob.glob(f"{volume}/*.pdf"):
    pdfs.append(file)

print(pdfs)

# COMMAND ----------

# MAGIC %md
# MAGIC Declare a function to run [`unstructured`](https://unstructured.io/) paritioning and chunking functions. 

# COMMAND ----------

import pandas
import datetime as dt

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

from pyspark.sql.types import ArrayType, StringType

def parse_pdfs(location: str) -> list:
  stime = dt.datetime.now()
  # elements = partition_pdf(location, strategy="ocr_only")
  elements = partition_pdf(location, strategy="hi_res")
  chunks = chunk_by_title(elements)
  print(str(dt.datetime.now() - stime) + f" for {location}")
  return [str(x) for x in chunks if len(str(x)) > 50]


# COMMAND ----------

all_chunks = []
for i in range(len(pdfs)):
  chunks = parse_pdfs(pdfs[i])
  all_chunks = all_chunks + [{
                                "chunk_id": f"{str(i)}_{str(j)}",
                                "file": pdfs[i],
                                "chunk": x
                             } for j,x in enumerate(chunks)]

# COMMAND ----------

# MAGIC %md
# MAGIC Take the chunking results and write them to a Delta table.

# COMMAND ----------

chunk_table = (spark.createDataFrame(all_chunks)
                    .select(['chunk_id', 'file', 'chunk'])
                    .write
                    .mode("overwrite")
                    .saveAsTable("pdf_chunks")
)

spark.sql('ALTER TABLE pdf_chunks SET TBLPROPERTIES (delta.enableChangeDataFeed = true)')


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM pdf_chunks;

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient(disable_notice=True)

VECTOR_SEARCH_ENDPOINT_NAME = "one-env-shared-endpoint-7"

try:
  vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")
except Exception as e: 
  print(e)

print(f"\nEndpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

#The table we'd like to index
source_table_fullname = f"{catalog}.{db}.pdf_chunks"
# Where we want to store our index
vs_index_fullname = f"{catalog}.{db}.pdf_chunks_vs_index"

print(f"Creating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
vsc.create_delta_sync_index(
  endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
  index_name=vs_index_fullname,
  source_table_name=source_table_fullname,
  pipeline_type="TRIGGERED",
  primary_key="chunk_id",
  embedding_source_column='chunk', #The column containing our text
  embedding_model_endpoint_name='databricks-gte-large-en' #The embedding endpoint used to create the embeddings
)

print(f"\nindex {vs_index_fullname} on table {source_table_fullname} is ready")

# COMMAND ----------

# MAGIC %md
# MAGIC You may have to wait a couple of minutes for the embedding model to populate the vectors on the vector search table.
# MAGIC
# MAGIC But then you can query the table with vector search!

# COMMAND ----------

question = "How many amendments are there in the Ohio state plan?"
results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).similarity_search(
  query_text=question,
  columns=["file", "chunk"],
  num_results=5)
docs = results.get('result', {}).get('data_array', [])
docs

# COMMAND ----------


