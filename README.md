# ParlaBERT

ParlaBERT is an NLP project that fine-tunes BERT on Swiss political speech to
predict the political party associated with a given statement.

### Dataset

This dataset contains cleaned speeches from the Swiss Federal Parliament from 
1999 to 2026. The speeches are sourced directly from the Parliamentary Services
web services and linked to the political party each speaker belonged to at the
time of the speech.

The dataset was cleaned by removing non-speech entries, government and chairing
roles, rapporteur statements, very short speeches, duplicate records
and entries with missing or unusable metadata.