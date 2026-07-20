# Portions derived from AutoRAG (https://github.com/Marker-Inc-Korea/AutoRAG), Apache-2.0.
# Modified by the RAG-Stack authors for namespace and runtime integration; see LICENSE.autorag and NOTICE.

import os
from typing import List

from rag_stack.static_rag_evaluator.support import dynamically_find_function
from rag_stack.static_rag_evaluator.utils.util import load_yaml_config
from rag_stack.static_rag_evaluator.vectordb.base import BaseVectorStore


# Vectordb-config keys that are never hyperparameters (metadata / identifiers /
# derived-from-data). The search-space builder skips these when scanning a
# vectordb config block for sweepable fields. `embedding_model` is
# INTENTIONALLY not included — written as a list it becomes a search dim, and
# the FAISS path resolver keys on it so different models live in separate
# index directories.
VECTORDB_SKIP_KEYS = frozenset({
	"name", "db_type", "collection_name", "path",
	"N",
})


SUPPORT_VECTORDB = {
	"chroma": ("rag_stack.static_rag_evaluator.vectordb.chroma", "Chroma"),
	"Chroma": ("rag_stack.static_rag_evaluator.vectordb.chroma", "Chroma"),
	"milvus": ("rag_stack.static_rag_evaluator.vectordb.milvus", "Milvus"),
	"Milvus": ("rag_stack.static_rag_evaluator.vectordb.milvus", "Milvus"),
	"weaviate": ("rag_stack.static_rag_evaluator.vectordb.weaviate", "Weaviate"),
	"Weaviate": ("rag_stack.static_rag_evaluator.vectordb.weaviate", "Weaviate"),
	"pinecone": ("rag_stack.static_rag_evaluator.vectordb.pinecone", "Pinecone"),
	"Pinecone": ("rag_stack.static_rag_evaluator.vectordb.pinecone", "Pinecone"),
	"couchbase": ("rag_stack.static_rag_evaluator.vectordb.couchbase", "Couchbase"),
	"Couchbase": ("rag_stack.static_rag_evaluator.vectordb.couchbase", "Couchbase"),
	"qdrant": ("rag_stack.static_rag_evaluator.vectordb.qdrant", "Qdrant"),
	"Qdrant": ("rag_stack.static_rag_evaluator.vectordb.qdrant", "Qdrant"),
	"faiss_ivf": ("rag_stack.static_rag_evaluator.vectordb.faiss_ivf", "FaissIVF"),
	"FaissIVF": ("rag_stack.static_rag_evaluator.vectordb.faiss_ivf", "FaissIVF"),
	"faiss_hnsw": ("rag_stack.static_rag_evaluator.vectordb.faiss_hnsw", "FaissHNSW"),
	"FaissHNSW": ("rag_stack.static_rag_evaluator.vectordb.faiss_hnsw", "FaissHNSW"),
}


def get_support_vectordb(vectordb_name: str):
	return dynamically_find_function(vectordb_name, SUPPORT_VECTORDB)


def load_vectordb(vectordb_name: str, **kwargs):
	vectordb = get_support_vectordb(vectordb_name)
	return vectordb(**kwargs)


def load_vectordb_from_yaml(
	yaml_path: str,
	vectordb_name: str,
	project_dir: str,
	read_only: bool = False,
):
	config_dict = load_yaml_config(yaml_path)
	vectordb_list = config_dict.get("vectordb", [])
	if len(vectordb_list) == 0 or vectordb_name == "default":
		chroma_path = os.path.join(project_dir, "resources", "chroma")
		return load_vectordb(
			"chroma",
			client_type="persistent",
			embedding_model="openai",
			collection_name="openai",
			path=chroma_path,
		)

	target_dict = list(filter(lambda x: x["name"] == vectordb_name, vectordb_list))
	target_dict[0].pop("name")  # delete a name key
	target_vectordb_name = target_dict[0].pop("db_type")
	target_vectordb_params = target_dict[0]
	if target_vectordb_name in ("faiss_ivf", "FaissIVF", "faiss_hnsw", "FaissHNSW"):
		target_vectordb_params["read_only"] = read_only
	return load_vectordb(target_vectordb_name, **target_vectordb_params)


def load_all_vectordb_from_yaml(
	yaml_path: str, project_dir: str
) -> List[BaseVectorStore]:
	config_dict = load_yaml_config(yaml_path)
	vectordb_list = config_dict.get("vectordb", [])
	if len(vectordb_list) == 0:
		chroma_path = os.path.join(project_dir, "resources", "chroma")
		return [
			load_vectordb(
				"chroma",
				client_type="persistent",
				embedding_model="openai",
				collection_name="openai",
				path=chroma_path,
			)
		]

	result_vectordbs = []
	for vectordb_dict in vectordb_list:
		_ = vectordb_dict.pop("name")
		vectordb_type = vectordb_dict.pop("db_type")
		vectordb = load_vectordb(vectordb_type, **vectordb_dict)
		result_vectordbs.append(vectordb)
	return result_vectordbs
