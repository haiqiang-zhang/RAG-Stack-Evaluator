from datetime import datetime
from typing import Dict, List, Tuple

from llama_index.core.schema import NodeRelationship


def add_essential_metadata(metadata: Dict) -> Dict:
	if "last_modified_datetime" not in metadata:
		metadata["last_modified_datetime"] = datetime.now()
	return metadata


def add_essential_metadata_llama_text_node(metadata: Dict, relationships: Dict) -> Dict:
	if "last_modified_datetime" not in metadata:
		metadata["last_modified_datetime"] = datetime.now()

	if "prev_id" not in metadata:
		if NodeRelationship.PREVIOUS in relationships:
			prev_node = relationships.get(NodeRelationship.PREVIOUS, None)
			if prev_node:
				metadata["prev_id"] = prev_node.node_id

	if "next_id" not in metadata:
		if NodeRelationship.NEXT in relationships:
			next_node = relationships.get(NodeRelationship.NEXT, None)
			if next_node:
				metadata["next_id"] = next_node.node_id
	return metadata


def get_start_end_idx(original_text: str, search_str: str) -> Tuple[int, int]:
	start_idx = original_text.find(search_str)
	if start_idx == -1:
		return 0, 0
	end_idx = start_idx + len(search_str)
	return start_idx, end_idx - 1
