# Task 2: Named Entity Recognition (NER)

https://absorbing-action-f33.notion.site/Classification-NER-Model-1e3094bc1d758025b726ce8554651b3b


### Setup instructions

#### Local Windows

1. Run the following in your terminal as an Administrator if you don't have LongPathsEnabled

```
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

2. Run `python -m pip install -r requirements.txt`
3. Run the notebook: `ner_v2.ipynb`


### Brief description of your approach

I started by looking at the data for 1 sentenec: 

sentence_id	word	tag
1	Patients	O
1	experienced	O
1	cough	B-SYMPTOM
1	after	O
1	administration	O
1	of	O
1	75mg	B-DOSAGE
1	of	O
1	DrugZ	O



I started by looking at the data, realizing that Drug[letter] is NOT an entity, and then replacing all instances with a random drug name and updating the tag to B-Drug.

I backtracked and lookd at unique entity examples and realize it's the same 8 symptoms and same 8 doses. I removed duplicates and asked DeepSeek to generate a sample notebook for the spaCy and transformer approach. The base models did not  recognize diahreea as a symptom so I decided to generate more data with the help of deepseek.

I was leaning towarsd spacy since it's simpler, but a domain specific transformer may have better results. DeepSeek specifically mentioned `BioClinicalBERT` when I was generated documentation for task 1. I decided to google further since this task seems common. I found https://www.youtube.com/watch?v=xpiDPdBpS18 which led me to a finetuned NER model that recognized bio-medical entities. The model was built ontop of `distilbert-base-uncased`. Instead of trying to re-invent the wheel, I tried to understand the existing [research paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9931203/) behind this model since it accurately identified the entities requested (sympton, dosage, drug).
- Paper: [Large-scale application of named entity recognition to biomedicine and epidemiology (Raza 2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9931203/)

![bme-ner-all-HF accurately recognizes entities from: 'Patients experienced cough after administration of 75mg of DrugZ'](bme-ner-all-HF.png).
[SOURCE](https://huggingface.co/d4data/biomedical-ner-all?text=Patients+experienced+cough+after+administration+of+75mg+of+DrugZ)


I attempt to recreate the model in this research paper. Interestingly enough, they compare their metrics against ClinicalBERT which is in the domain of pre-trained models I was planning on finetuning for this task. They achieved an F1 score ~10% higher than ClinicalBERT.

My personal computer struggles to handle the training (~30m on a much more powerful GPU than mine), so I aim to get ~80% F1 score.


### Training and evaluation guide

The DeepSeek generated graph seemed decent enough given the training data. I purposefully kept only the drugs and symptoms that were 1 word to avoid using the INSIDE tag. The default parameters seemed fine given the shallow training data.

I also used example sentences with edge cases to see if my model could extend to words it was NOT trained on.

#### Hyperparameter Tuning Strategy

I adjusted parameters based on the training and validation loss graph. 
I read the paper [Large-scale application of named entity recognition to biomedicine and epidemiology (Raza 2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9931203/). It's trained a different data so I can't apply the same hyperparameters. It mentioned warmup steps but since I'm using early stopping

I used early stopping because although there is a lot of data, it's a lot of the 'same' data, so it's like having 'limited data'. Thus, I used early stopping to prevent overfitting.

Increased epochs to 10, set early stopping with patience 3. The results were pretty much the same, so I just returned ack to notebook v2 (simpler)

#### Validation Methodology

#### Diagnostic Tools:


**Training Arguments v2:**
```python
training_args = TrainingArguments(
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)
```

**Training Arguments v3 (early stopping):**
```python
training_args = TrainingArguments(
    output_dir="/tmp/ner_results_v3",
    eval_strategy="steps",
    logging_strategy="steps",

    eval_steps=10, 
    logging_steps=10,

    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10, # increase epochs since early stopping implemented
    weight_decay=0.01,
    logging_dir="/tmp/ner_logs",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)
```


### Results

- Entity-level evaluation metrics (F1, precision, recall)

Entity-Level Metrics by Type:
  DRUG:
    Precision: 0.9962
    Recall: 1.0000
    F1-Score: 0.9981
    True entities: 265
    Predicted entities: 266
  SYMPTOM:
    Precision: 1.0000
    Recall: 1.0000
    F1-Score: 1.0000
    True entities: 265
    Predicted entities: 265
  DOSAGE:
    Precision: 1.0000
    Recall: 1.0000
    F1-Score: 1.0000
    True entities: 265
    Predicted entities: 265

Key findings:
- 

### Examples

- Show model predictions on at least 5 example sentences


==================================================
Input Text: 500mg Tylenol every 6h for fever, but dizziness occurred

Predicted Entities:
- DOSAGE: '500mg' (positions 0-1)
- DRUG: 'Tylenol' (positions 1-2)
- DOSAGE: '6h' (positions 3-4)
- SYMPTOM: 'fever,' (positions 5-6)
- SYMPTOM: 'dizziness' (positions 7-8)
==================================================

==================================================
Input Text: Patients experienced cough after administration of 75mg of Multaq

Predicted Entities:
- SYMPTOM: 'cough' (positions 2-3)
- DOSAGE: '75mg' (positions 6-7)
- DRUG: 'Multaq' (positions 8-9)
==================================================

==================================================
Input Text: Aspirin 81mg daily caused GI bleeding and tinnitus

Predicted Entities:
- DRUG: 'Aspirin' (positions 0-1)
- DOSAGE: '81mg' (positions 1-2)
- SYMPTOM: 'GI' (positions 4-5)
- SYMPTOM: 'bleeding' (positions 5-6)
- DRUG: 'tinnitus' (positions 7-8)
==================================================

==================================================
Input Text: Overdose on CaCO3 (calcium carbonate): vomiting, drowsiness

Predicted Entities:
- DRUG: 'CaCO3' (positions 2-3)
- DRUG: 'carbonate):' (positions 4-5)
- SYMPTOM: 'vomiting,' (positions 5-6)
- SYMPTOM: 'drowsiness' (positions 6-7)
==================================================

==================================================
Input Text: Lisinoprol 10mg led to dry cough and fatigue

Predicted Entities:
- DRUG: 'Lisinoprol' (positions 0-1)
- DOSAGE: '10mg' (positions 1-2)
- SYMPTOM: 'dry' (positions 4-5)
- SYMPTOM: 'cough' (positions 5-6)
- SYMPTOM: 'fatigue' (positions 7-8)
==================================================

==================================================
Input Text: Janumet XR 50mg/1000mg BID caused diarrhea

Predicted Entities:
- DRUG: 'Janumet' (positions 0-1)
- DRUG: 'XR' (positions 1-2)
- DOSAGE: '50mg/1000mg' (positions 2-3)
- DRUG: 'BID' (positions 3-4)
- SYMPTOM: 'diarrhea' (positions 5-6)
==================================================

==================================================
Input Text: Street drug 'Molly' induced hyperthermia and seizures

Predicted Entities:
- SYMPTOM: 'hyperthermia' (positions 4-5)
- SYMPTOM: 'seizures' (positions 6-7)
==================================================

==================================================
Input Text: No ibuprofen use, but naproxen 250mg caused dyspepsia

Predicted Entities:
- DRUG: 'ibuprofen' (positions 1-2)
- DRUG: 'naproxen' (positions 4-5)
- DOSAGE: '250mg' (positions 5-6)
- SYMPTOM: 'dyspepsia' (positions 7-8)
==================================================