# NER Switzerland (Multilingual NER)

This project explores **multilingual Named Entity Recognition (NER)** for the Swiss context by mixing the main Swiss languages: German (`de`), French (`fr`), Italian (`it`), and English (`en`).

So far, the work focuses on **dataset discovery, Swiss-proportion sampling, and basic tag inspection** using Hugging Face Datasets.

## What's implemented so far

All progress lives in: `Multilingal_NER/Ner_Switzerland.ipynb`.

Completed steps:

1. Installed and used the Hugging Face `datasets` library.
2. Discovered available XTREME subsets via `google/xtreme`.
3. Focused on the `PAN-X.*` multilingual NER subsets.
4. Loaded PAN-X data for `de`, `fr`, `it`, and `en`.
5. Built an imbalanced Swiss-style dataset by **downsampling each split** according to spoken proportions: German `0.629`, French `0.229`, Italian `0.084`, English `0.059`.
6. Stored the result in a `defaultdict(DatasetDict)` named `panx_ch`.
7. Inspected a German training example (tokens and `ner_tags`).
8. Examined dataset features and the `ClassLabel` mapping for `ner_tags`.
9. Added a readable `ner_tags_str` column using `ClassLabel.int2str()` and `map()`.
10. Displayed token/tag alignment for a sample and computed entity-type frequencies (B- tags) across train/validation/test splits.

## Current sampled training sizes

From the notebook output:

| Split | de | fr | it | en |
| --- | ---: | ---: | ---: | ---: |
| train examples | 12,580 | 4,580 | 1,680 | 1,180 |

## Entity tag counts (German split)

Counts of B- tags from `panx_de`:

| Split | LOC | ORG | PER |
| --- | ---: | ---: | ---: |
| train | 6,186 | 5,366 | 5,810 |
| validation | 3,172 | 2,683 | 2,893 |
| test | 3,180 | 2,573 | 3,071 |

## Modeling direction

- Chosen backbone: **XLM-R** (multilingual transformer; SentencePiece tokenizer trained on 100 languages).
- Tokenizer check: compared WordPiece (`bert-base-cased`) vs SentencePiece (`xlm-roberta-base`) on `"Jack sparrow loves New York!"`.
  - BERT tokens: `[CLS], Jack, spa, ##rrow, loves, New, York, !, [SEP]`
  - XLM-R tokens: `<s>, ▁Jack, ▁spar, row, ▁love, s, ▁New, ▁York, !, </s>`
- Custom head: `XLMRobertaForTokenClassification` subclass adds dropout + linear classifier on top of `RobertaModel`; uses `CrossEntropyLoss` ignoring padded labels (`-100`); returns `TokenClassifierOutput`.

## How to run

Open and run the notebook:

- `Multilingal_NER/Ner_Switzerland.ipynb`

Minimal dependencies used so far:

```bash
pip install datasets pandas
```
