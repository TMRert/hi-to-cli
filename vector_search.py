# Databricks notebook source
# MAGIC %pip install databricks-vectorsearch
# MAGIC dbutils.library.restartPython()
# MAGIC
# MAGIC

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

client = VectorSearchClient()

# COMMAND ----------

# create Vector Search Endpoint

client.create_endpoint(
    name="dais_hackathon",
    endpoint_type="STANDARD"
)

# COMMAND ----------

# Create Vector Search Index

index = client.create_delta_sync_index(
  endpoint_name="dais_hackathon",
  source_table_name="workspace.default.man_7_info",
  index_name="workspace.default.man_7_info_index",
  pipeline_type='TRIGGERED',
  primary_key="Command",
  embedding_source_column="Text",
  embedding_model_endpoint_name="databricks-gte-large-en"
)

# COMMAND ----------

# Delta Sync Index with embeddings computed by Databricks

results = index.similarity_search(
    query_text="How do I create a connection to other machine",
    columns=["Text", "Summary"],
    num_results=2
    )

results   
