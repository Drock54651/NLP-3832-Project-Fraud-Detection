# NLP-3832-Project-Fraud-Detection
3832 NLP Semester Project

Dataset Links (Too big for Github):
- Labelled Fake Job Postings: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
- LinkedIn Job Postings: https://www.kaggle.com/datasets/arshkon/linkedin-job-postings
- SMS Spam: https://huggingface.co/datasets/ucirvine/sms_spam
- Email Spam: https://www.kaggle.com/datasets/meruvulikith/190k-spam-ham-email-dataset-for-classification

How to reproduce:
The above datasets will need to be downloaded except for the Huggingface dataset. All datasets must be present in the Data directory.
Please ensure the datasets have the following names:
Fake_Real_Job_Posting.csv for EMSCAD
spam_Emails_data.csv for the Email dataset
linkedin_postings.csv for the linkedin set
indeed-job-listings-information.csv for the Indeed set


Run all cells in lab. If you experience memory issues run all cells except the cell training the new model on email only , testing the model on emails only, and the graphing cells at the end of the notebook. After doing this, run the "del model_sms" cell, then run the model model running only email. After this, run the graphing cells.


Do not run the graphing cells at the end of the notebook before training and test all models, this will cause errors.

The following libraries are required and may need to be installed: Pytorch, Pandas, Numpy, Seaborn, Nltk, huggingface datasets, tqdm, pytorch-transformers


The data preprocessing is done in the notebook. It shows the distributions in data, merges categorical columns into 1 text field, and includes cleaning functions to remove white spaces, numbers, and hyperlinks. Additionally, the input was tokenized using a pre-trained BERT tokenizer. (You should be able to get the data preprocessed by just running the notebook).

Model's hyperparameters:
- MODEL_NAME = "bert-base-uncased"
- BATCH_SIZE = 16
- MAX_LEN    = 128
- EPOCHS     = 10
- LR         = 2e-5

After initial fine-tuning on the job postings dataset:
- Expected test set performance for job postings: accuracy - 98%, F-1 score - 
- Expected test set performance for unlabeled Indeed postings:
  - soft-max derived probabilities distribution: mostly within 80-100% for predictions
- Expected test set performance for unlabeled LinkedIn postings:
  - softmax derived probabilities distribution: mostly within 80-100% for predictions
 - Expected test set performance for SMS spam: accuracy - , F-1 score - 
 - Expected test set performance for email spam: accuracy - , F-1 score -
   
Examples of what the model labeled as fraudulent in the unlabeled data:

After additional fine-tuning on the SMS and email datasets: accuracy - , F-1 score - 
- Expected test set performance for job postings: accuracy - , F-1 score - 
- Expected test set performance for unlabeled Indeed postings: 
- Expected test set performance for unlabeled LinkedIn postings: 

Work distribution:
- Declan:  fine-tuning the BERT model on SMS and email
- Derick:  unlabeled data scraping
- John: fine-tuning the BERT model on the job postings dataset
- Taya: all the data pre-processing and tokenization
