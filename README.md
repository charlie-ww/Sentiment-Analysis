# Sentiment-Analysis

## Install dependencies
```bash
pip install -r requirements.txt
```
## Create the dictionary
```bash
python create_dictionary.py
```
If not work try:
```bash
python3 create_dictionary.py
```
## Test tokenizer
```bash
python test_tokenizer.py
```
If not work try:
```bash
python3 test_tokenizer.py
```
You should see the following output:
```bash
[Text]: Embrace self-love and simplicity in life.
[Token_ids]: [1, 15078, 36550, 4766, 37451, 22455, 25444, 2, 0, 0, 0, 0, 0, 0, 0, 0]
[Decode_text]: [BOS] embrace self-love and simplicity in life. [EOS] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]

[Text]: The app offers a wide range of features, which is impressive.
[Token_ids]: [1, 40868, 5091, 29651, 3344, 44931, 33646, 29604, 16748, 44842, 23481, 22415, 2, 0, 0, 0]
[Decode_text]: [BOS] the app offers a wide range of features, which is impressive. [EOS] [PAD] [PAD] [PAD]
```
## Train the model
```bash
python train.py
```
If not work try:
```bash
python3 train.py
```
You should see the training process in the terminal.
And the model will be saved as `model.pth` in the current directory.
Then the prediction will be saved as `predictions.csv` in the current directory.
## Submit the prediction
```bash
kaggle competitions submit -c 2024-introduction-to-ai-hw-4-sentiment-analysis -f predictions.csv -m "Message"