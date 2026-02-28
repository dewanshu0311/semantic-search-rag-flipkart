# Manual Evaluation Instructions

## Purpose
Label retrieved documents for relevance so we can compute real Precision@K and Recall@K values.

## Steps

1. Launch the labelling UI:
   ```bash
   streamlit run app/manual_eval_app.py
   ```

2. Enter a search query in the text box.
3. Review the top-K results presented.
4. For each result, click **Relevant** or **Not Relevant**.
5. Click **Save Labels** to write your annotations to `evaluation/manual_labels.csv`.
6. Repeat for 10-20 queries to build a meaningful evaluation set.

## Label Format (manual_labels.csv)

| query | doc_id | relevant |
|-------|--------|----------|
| healthy snack | 42 | 1 |
| healthy snack | 87 | 0 |

## Computing Metrics

```python
from evaluation.eval_metrics import precision_at_k, recall_at_k
import pandas as pd

labels = pd.read_csv("evaluation/manual_labels.csv")
# Group by query and compute metrics
```
