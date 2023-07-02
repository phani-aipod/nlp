import numpy as np
import pandas as pd
import re
import torch
import random
import torch.nn as nn
import transformers

# specify GPU
import torch
#device = torch.device("cuda")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# used a dictionary to represent an intents JSON file
data = {"intents": [
{"tag": "profile",
 "responses": ["Profiles determine the level of access a user can have in a Salesforce org.", "One profile can be assigned to any number of users. Take the example of a Sales or Service team in a company. The entire team will be assigned the same profile. The admin can create one profile: Sales Profile, which will have access to the Leads, Opportunities, Campaigns, Contacts and other objects deemed necessary by the company.",   "each user can only be assigned 1 profile."]},
{"tag": "governor",
 "responses": ["Governor Limits are a Salesforce developer’s biggest challenge. That is because if the Apex code ever exceeds the limit, the expected governor issues a run-time exception that cannot be handled. Hence as a Salesforce developer, you have to be very careful while developing your application."]},
{"tag": "sandbox",
 "responses": ["A sandbox is a copy of the production environment/ org, used for testing and development purposes. It’s useful because it allows development on Apex programming without disturbing the production environment."]},
{"tag": "apex",
 "responses": ["No, it is not possible to edit apex classes and triggers directly in production environment."]},
{"tag": "fieldrecord",
 "responses": ["A standard field record name can have data type of either auto number or text field with a limit of 80 chars."]}
]}

# We have prepared a chitchat dataset with 5 labels
df = pd.read_excel("chitchat.xlsx")
#df.head()

# Converting the labels into encodings
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
# check class distribution
df['label'].value_counts(normalize = True)


max_seq_len = 8

class BERT_Arch(nn.Module):
   def __init__(self, bert):      
       super(BERT_Arch, self).__init__()
       self.bert = bert 
      
       # dropout layer
       self.dropout = nn.Dropout(0.2)
      
       # relu activation function
       self.relu =  nn.ReLU()
       # dense layer
       self.fc1 = nn.Linear(768,512)
       self.fc2 = nn.Linear(512,256)
       self.fc3 = nn.Linear(256,5)
       #softmax activation function
       self.softmax = nn.LogSoftmax(dim=1)
       #define the forward pass
   def forward(self, sent_id, mask):
      #pass the inputs to the model  
      cls_hs = self.bert(sent_id, attention_mask=mask)[0][:,0]
      
      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)
      
      x = self.fc2(x)
      x = self.relu(x)
      x = self.dropout(x)
      # output layer
      x = self.fc3(x)
   
      # apply softmax activation
      x = self.softmax(x)
      return x

import re
import random
import torch
import numpy as np

def get_prediction(text):
    text = re.sub(r'[^a-zA-Z ]+', '', text)
    test_text = [text]
    model.eval()

    tokens_test_data = tokenizer(
        test_text,
        max_length=max_seq_len,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )
    test_seq = torch.tensor(tokens_test_data['input_ids'])
    test_mask = torch.tensor(tokens_test_data['attention_mask'])

    preds = None
    with torch.no_grad():
        preds = model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().cpu().numpy()
    preds = np.argmax(preds, axis=1)
    predicted_intent = le.inverse_transform(preds)[0]
    print("Intent Identified:", predicted_intent)
    return predicted_intent

def get_response(message):
    intent = get_prediction(message)
    for i in data['intents']:
        if i["tag"] == intent:
            result = random.choice(i["responses"])
            break
    print(f"Response: {result}")
    return "Intent: " + intent + '\n' + "Response: " + result

from transformers import DistilBertTokenizer, DistilBertModel
# Load the DistilBert tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

#load the saved model
model = torch.load('trained_model.pth',torch.device(device))

from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return get_response(userText)
if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
