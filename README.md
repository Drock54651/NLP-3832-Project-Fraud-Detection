# NLP-3832-Project-Fraud-Detection
3832 NLP Semester Project

Dataset Links (Too big for Github):
- Labelled Fake Job Postings: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
- LinkedIn Job Postings: https://www.kaggle.com/datasets/arshkon/linkedin-job-postings
- SMS Spam: https://huggingface.co/datasets/ucirvine/sms_spam
- Email Spam: https://www.kaggle.com/datasets/meruvulikith/190k-spam-ham-email-dataset-for-classification

## Important Note About Branches 
We decided to divide certain aspects of our projects via branches.
- The main branch
  - Responsible for the initial data preprocessing and is the foundation for the other 2 branches.
- The Evaluation branch
  - Introduces the unlabeled LinkedIn and Indeed datasets, and consists of having our trained model predict on the 2 unlabeled dataset in addition to the email and sms test sets.
- The Fine-tuning branch
  - Includes making adjustments to our model and also displays additional metrics like the f1-scores.

Overall you should mainly be switching between the Evaluation and Fine-tuning branch if you want to see our results.

-----
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
- EPOCHS     = 6
- LR         = 2e-5

After initial fine-tuning on the job postings dataset:
- Expected test set performance for job postings: accuracy - 98%, F-1 score - 85%
- Expected test set performance for unlabeled Indeed postings:
  - soft-max derived probabilities distribution: mostly within 80-100% for predictions
- Expected test set performance for unlabeled LinkedIn postings:
  - softmax derived probabilities distribution: mostly within 80-100% for predictions
 - Expected test set performance for SMS spam: accuracy - , F-1 score -
 - Expected test set performance for email spam: accuracy - , F-1 score -
  

Note: For the unlabeled datasets, we did not have any way of discerning some kind of ground truth for them. Instead, they were used as a way to see what our already trained model would label them as. After doing so, a human could extract the postings labeled as fraudulent by the model and the human would have to check if those postings are truely fraudulent or not.

Examples of what the model labeled as fraudulent in the unlabeled data:
  - Indeed: [Example 1](/images/Indeed/Screenshot%202025-05-05%20205621.png), [Example 2](/images//Indeed/Screenshot%202025-05-05%20205833.png)
  - LinkedIn: [Example 1](/images/LinkedIn/Screenshot%202025-05-05%20205037.png), [Example 2](/images/LinkedIn/Screenshot%202025-05-05%20205354.png)

Example of what the model labeled as fraudulent in the labeled SMS and Email data:
  - SMS: [Example](/images/SMS/Screenshot%202025-05-05%20210058.png)
  - Email: [Example](/images/Email/Screenshot%202025-05-05%20210304.png)
  - 
After additional finetuning on SMS:
- Expected test set performance for job postings: accuracy - 97%, F-1 score - 85%
- Expected test set performance for spam email: accuracy - 49%, F-1 score - 55%
- Expected test set performance for spam sms: accuracy - 99%, F-1 score - 97%

After additional fine-tuning on the SMS and email datasets: 
- Expected test set performance for job postings: accuracy - , F-1 score -
- Expected test set performance for unlabeled Indeed postings: 
- Expected test set performance for unlabeled LinkedIn postings: 

Work distribution:
- Declan:  fine-tuning the BERT model on SMS and email
- Derick:  model evaluation
- John: fine-tuning the BERT model on the job postings dataset
- Taya: all the data pre-processing and tokenization
