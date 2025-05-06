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

The following libraries are required and may need to be installed:
Pytorch
Pandas
Numpy
Seaborn
Nltk
huggingface datasets
tqdm
pytorch-transformers


All hyperparameters used and all data preprocessing is in the Notebook.
