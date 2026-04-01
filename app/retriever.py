import json

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# TF-IDF Baseline QA System
class TFIDFBaseline:
	"""Simple retriever baseline that ranks contexts by TF-IDF cosine similarity."""

	def __init__(self):
		# Use unigram + bigram features to capture short phrase-level signals.
		self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
		self.context_matrix = None
		self.contexts = []

	def fit(self, contexts):
		"""Fit the TF-IDF vectorizer on all contexts and cache the matrix."""
		self.contexts = contexts
		self.context_matrix = self.vectorizer.fit_transform(contexts)

	def predict(self, question):
		"""Return the most similar context and its cosine similarity score."""
		question_vec = self.vectorizer.transform([question])
		similarities = cosine_similarity(question_vec, self.context_matrix)[0]
		best_idx = int(np.argmax(similarities))
		return self.contexts[best_idx], float(similarities[best_idx])


# Prepare PubMedQA contexts and answers
def prepare_data(dataset):
	"""Extract flattened contexts, long answers, and questions from train split."""
	contexts = []
	answers = []
	questions = []

	for example in dataset["train"]:
		context = " ".join(example["context"]["contexts"])
		answer = example["long_answer"]
		question = example["question"]

		contexts.append(context)
		answers.append(answer)
		questions.append(question)

	return {"contexts": contexts, "answers": answers, "questions": questions}


# Word overlap F1 score
def compute_f1(prediction, ground_truth):
	"""Compute set-based overlap F1 between prediction and reference text."""
	pred_words = set(prediction.lower().split())
	truth_words = set(ground_truth.lower().split())

	if not pred_words or not truth_words:
		return 0.0

	overlap = pred_words.intersection(truth_words)
	if not overlap:
		return 0.0

	precision = len(overlap) / len(pred_words)
	recall = len(overlap) / len(truth_words)

	if precision + recall == 0:
		return 0.0
	return 2 * precision * recall / (precision + recall)


# Evaluate TF-IDF baseline
def evaluate_baseline(baseline, data, n_samples=100):
	"""Evaluate average word-overlap F1 on the first n_samples examples."""
	sample_count = min(n_samples, len(data["questions"]))
	f1_scores = []

	for i in range(sample_count):
		question = data["questions"][i]
		ground_truth = data["answers"][i]

		predicted_context, _ = baseline.predict(question)
		f1_scores.append(compute_f1(predicted_context, ground_truth))

	return float(np.mean(f1_scores)) if f1_scores else 0.0


if __name__ == "__main__":
	# Load PubMedQA and build retrieval-ready data structures.
	dataset = load_dataset("pubmed_qa", "pqa_labeled")
	data = prepare_data(dataset)

	# Create and fit TF-IDF retriever on all train contexts.
	baseline = TFIDFBaseline()
	baseline.fit(data["contexts"])

	# Show retrieval behavior on the first 3 questions.
	for i in range(3):
		question = data["questions"][i]
		top_context, score = baseline.predict(question)

		print(f"\nExample {i + 1}")
		print(f"Question: {question}")
		print(f"Top retrieved context: {top_context[:200]}...")
		print(f"Similarity score: {score:.4f}")

	# Evaluate the baseline on 100 samples with overlap F1.
	baseline_f1 = evaluate_baseline(baseline, data, n_samples=100)
	print(f"Baseline F1 Score: {baseline_f1:.2f}")
	print("TF-IDF Baseline evaluation complete!")

	# Keep json import exercised while staying side-effect free.
	_ = json.dumps({"baseline_f1": round(baseline_f1, 4)})
