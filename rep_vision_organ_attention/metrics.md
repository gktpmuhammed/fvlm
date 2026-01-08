## Traditional N-Gram Metrics (Lexical Overlap)
These metrics measure word-for-word similarity between your generated report and the ground truth (the radiologist's actual report). They focus on "fluency."

### BLEU-1, -2, -3, -4 (Bilingual Evaluation Understudy)
What it measures: The precision of N-grams (sequences of 
N
N
 words).
BLEU-1: Individual words (unigrams). Measures if the keywords are present.

BLEU-4: Sequences of 4 words. Measures if whole phrases are correct.

Relation to Radiology:

High BLEU scores indicate your model is using the correct medical terminology and standard phrasing (e.g., "no acute cardiopulmonary abnormality").

Note: BLEU is often harsh; if a doctor writes "clear lungs" and your model writes "lungs are normal," BLEU might penalize this even though the medical meaning is identical.

### ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation)

What it measures: The Longest Common Subsequence (LCS). Unlike BLEU (precision), ROUGE focuses on recall.

Relation to Radiology:

It checks if the model captured all the information present in the reference.

It is robust to the structure of the sentence. If the reference is "Heart size normal," ROUGE-L rewards "The heart size is normal" highly.

### METEOR (Metric for Evaluation of Translation with Explicit ORdering)

What it measures: 

An improvement over BLEU that uses stemming and synonym matching (using WordNet).

Relation to Radiology:

This is crucial for medical text. It understands that "enlarged" and "cardiomegaly" are related, or "opacity" and "opacities" are the same word stem.
Observation from your table: Your METEOR scores (e.g., 0.428 for Heart) are much higher than your BLEU scores. This suggests your model is getting the meaning right, even if the exact wording varies.

### CIDEr (Consensus-based Image Description Evaluation)

What it measures: It uses TF-IDF (Term Frequency-Inverse Document Frequency) weighting. It gives low weight to common words (like "the", "is", "normal") and high weight to rare words.

Relation to Radiology:

This is extremely important. In radiology, the word "Pneumothorax" is rare but critical. "Normal" is common and less critical.

A high CIDEr score means your model is correctly identifying the rare, abnormal findings, which is the most important part of a diagnosis.

## Semantic Metrics (Meaning & Embedding)
These metrics use deep learning (BERT) to compare the meaning of sentences rather than just matching words.

### BERTScore

What it measures: 

It feeds both the generated text and reference text into a pre-trained BERT model and calculates the similarity of their embeddings.

Relation to Radiology:

It handles semantic equivalence.
Example: Ref: "No pleural effusion." | Pred: "Pleural spaces are clear."
Traditional metrics (BLEU) would give this a low score (no word overlap). BERTScore recognizes these mean the exact same thing medically.

## Clinical/Factual Correctness Metrics (The "Medical" Standard)

These are the most critical metrics for your thesis. They ignore grammar and focus purely on: "Is the diagnosis correct?"

### CheXbert

What it measures: 

It uses a BERT-based classifier trained to detect 14 specific chest pathologies (e.g., Atelectasis, Cardiomegaly, Pneumonia). It runs this classifier on your generation and the reference, then compares the results (F1-score).

Relation to Radiology:

This measures Diagnostic Accuracy.
If your CheXbert score is high, it means that if the patient had "Cardiomegaly," your model successfully reported "Cardiomegaly" (or enlarged heart).

### RadGraph

What it measures:

It converts the report into a Knowledge Graph (Entities + Relations).
Entity: "Opacity"
Relation: "Located in" -> "Left Lower Lobe"
Relation to Radiology:
It checks if the anatomical relationships are correct.
Standard text metrics might forgive moving a finding from the "left lung" to the "right lung." RadGraph punishes this heavily because it is a severe medical error.

### RadCliQ (Radiology Report Clinical Quality)

What it measures: A composite metric (combining BLEU, RadGraph) trained to predict how a human radiologist would rate the error.

Relation to Radiology:


It correlates best with human judgment. A high RadCliQ score implies a radiologist would find the report useful.

### GREEN (Generative Radiology Report Evaluation and Error Notation)

What it measures: 
A metric usually driven by a large language model trained to identify clinically significant errors (false positives/negatives) compared to the reference.

Relation to Radiology:

It acts like a "peer review." It penalizes hallucinations (making up diseases) more than grammatical errors.

### SRR-BERT (Semantic Re-ranking BERT)

What it measures: 

A metric that scores the semantic similarity specifically tuned on medical/radiology datasets.

Relation to Radiology:

Similar to BERTScore but fine-tuned to understand that "mild" and "severe" are antonyms in a medical context, whereas standard BERT might just see them as "adjectives."