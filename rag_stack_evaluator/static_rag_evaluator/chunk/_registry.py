# Portions derived from AutoRAG (https://github.com/Marker-Inc-Korea/AutoRAG), Apache-2.0.
# Modified by the RAG-Stack authors for namespace and runtime integration; see LICENSE.autorag and NOTICE.

import logging
from typing import List, Callable

from llama_index.core.node_parser import (
	TokenTextSplitter,
	SentenceSplitter,
	SentenceWindowNodeParser,
	SemanticSplitterNodeParser,
	SemanticDoubleMergingSplitterNodeParser,
	SimpleFileNodeParser,
)
from langchain_text_splitters import (
	RecursiveCharacterTextSplitter,
	CharacterTextSplitter,
	KonlpyTextSplitter,
	SentenceTransformersTokenTextSplitter,
)

from rag_stack_evaluator.static_rag_evaluator import LazyInit

logger = logging.getLogger("RAG-Stack")

chunk_modules = {
	# Llama Index
	"token": TokenTextSplitter,
	"sentence": SentenceSplitter,
	"sentencewindow": SentenceWindowNodeParser,
	"semantic_llama_index": SemanticSplitterNodeParser,
	"semanticdoublemerging": SemanticDoubleMergingSplitterNodeParser,
	"simplefile": SimpleFileNodeParser,
	# LangChain
	"sentencetransformerstoken": SentenceTransformersTokenTextSplitter,
	"recursivecharacter": RecursiveCharacterTextSplitter,
	"character": CharacterTextSplitter,
	"konlpy": KonlpyTextSplitter,
}


def split_by_sentence_kiwi() -> Callable[[str], List[str]]:
	try:
		from kiwipiepy import Kiwi
	except ImportError:
		raise ImportError(
			"You need to install kiwipiepy to use 'ko_kiwi' tokenizer. "
			"Please install kiwipiepy by running 'pip install kiwipiepy'. "
			"Or install Korean version of AutoRAG by running 'pip install AutoRAG[ko]'."
		)
	kiwi = Kiwi()

	def split(text: str) -> List[str]:
		kiwi_result = kiwi.split_into_sents(text)
		sentences = list(map(lambda x: x.text, kiwi_result))
		return sentences

	return split


sentence_splitter_modules = {"kiwi": LazyInit(split_by_sentence_kiwi)}
