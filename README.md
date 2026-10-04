# Subjectivity Mining, Assignment 4, Group 9

Offensive language detection with Logistic Regression, BERT, HateBERT and fBERT.
In-domain: trained on OLID train small, tested on OLID test.
Cross-domain: trained on HASOC train, tested on OLID test.

## Files
- report/A4-report-Group9.pdf: the report
- A4-notebook-Group9-FINAL.ipynb: notebook with all parts of the assignment
- olid-train-small.csv, hasoc-train.csv, olid-test.csv: the datasets (0 = non-offensive, 1 = offensive)
- block1_logreg: Logistic Regression predictions and scores (Ines Casado Vela)
- block2_bert: BERT predictions and scores (Helian X. Liu)
- block3_hatebert: HateBERT scripts, predictions, scores and logs (Martin Ranea Marina)
- block4_fbert: fBERT script, predictions, scores, logs, seed check and the script that compares all models (Pablos Tselioudis Garmendia)
- figures, tables, A4_all_model_metrics.csv: combined results of all models

## Main results (macro-F1 on the OLID test set)
| Model | In-domain | Cross-domain |
|---|---|---|
| Logistic Regression | 0.6923 | 0.5094 |
| BERT | 0.7813 | 0.7226 |
| HateBERT | 0.8018 | 0.7536 |
| fBERT | 0.8033 | 0.8088 |

## Running the code
The packages are listed in block3_hatebert/requirements.txt. Training the transformer models needs a GPU.
To train fBERT again: go to block4_fbert and run python fbert_block4.py --output-dir fbert_results_new --models-dir fbert_models_new
To make the combined tables and figures: go to block4_fbert and run python integrate_all.py
