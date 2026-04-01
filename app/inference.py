import json
import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


# Load BioBERT tokenizer
def load_tokenizer(model_name="dmis-lab/biobert-base-cased-v1.2"):
	"""Load and return an AutoTokenizer for the provided model name."""
	return AutoTokenizer.from_pretrained(model_name)


# Preprocess a single QA example
def preprocess_example(question, context, tokenizer, max_length=512, stride=128):
	"""Tokenize a question-context pair with overflow handling for long contexts."""
	tokenized = tokenizer(
		question,
		context,
		max_length=max_length,
		truncation="only_second",
		stride=stride,
		return_overflowing_tokens=True,
		return_offsets_mapping=True,
		padding="max_length",
		return_tensors="pt",
	)
	return tokenized


# Preprocess full PubMedQA dataset
def preprocess_dataset(dataset, tokenizer, max_length=512):
	"""Preprocess all examples and map final_decision labels to numeric targets."""
	label_map = {"no": 0, "yes": 1, "maybe": 2}
	preprocessed_examples = []

	for example in dataset:
		question = example["question"]
		context = " ".join(example["context"]["contexts"])
		tokenized = preprocess_example(
			question=question,
			context=context,
			tokenizer=tokenizer,
			max_length=max_length,
		)

		preprocessed_examples.append(
			{
				"question": question,
				"context": context,
				"inputs": tokenized,
				"label": label_map[example["final_decision"]],
				"final_decision": example["final_decision"],
			}
		)

	return preprocessed_examples


# Test the preprocessing pipeline
if __name__ == "__main__":
	# Keep imports explicitly present and confirm runtime availability.
	_ = json
	_ = os
	_ = torch

	dataset = load_dataset("pubmed_qa", "pqa_labeled")
	tokenizer = load_tokenizer()

	first_example = dataset["train"][0]
	question = first_example["question"]
	context = " ".join(first_example["context"]["contexts"])
	tokenized = preprocess_example(question, context, tokenizer)

	print(f"Question: {question}")
	print(f"Context length (characters): {len(context)}")
	print(f"input_ids shape: {tokenized['input_ids'].shape}")
	print(f"attention_mask shape: {tokenized['attention_mask'].shape}")
	print("Preprocessing pipeline working correctly!")
