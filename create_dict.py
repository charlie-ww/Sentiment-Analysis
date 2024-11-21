import json
import pickle


def create_dict(train_path, test_path, save_path):
    
    # 1. read datas
    datas = []
    with open(train_path, 'r') as f:
        for line in f.readlines():
            data = json.loads(line)                
            datas.append(data['text'])
    with open(test_path, 'r') as f:
        for line in f.readlines():
            data = json.loads(line)                
            datas.append(data['text'])
    print(f"Number of data = {len(datas)}")
                
                
    # 2. get all tokens that appear in the data
    all_tokens = set()
    for text in datas:
        text = text.lower() 
        # TODO: you can change the way to split the text if needed
        tokens = text.split(' ')
        all_tokens.update(tokens) 


    # 3. define special tokens, TODO: you can re-define special tokens if needed
    special_tokens = ['[PAD]', '[BOS]', '[EOS]', '[UNK]']


    # 4. create a dictionary that maps tokens to indices
    token_to_index = dict()
    token_id = 0
    for token in special_tokens + sorted([token for token in all_tokens if token not in special_tokens]):
        token_to_index[token] = token_id
        token_id += 1


    # 5. save the dictionary to a pickle file
    with open(save_path, "wb") as f:
        pickle.dump(token_to_index, f)
    print(f"Saved token to index dictionary at {save_path}.")


if __name__ == "__main__":
    # TODO: change the path to the dataset if needed
    train_path = "dataset/kaggle/train.jsonl"
    test_path = "dataset/kaggle/test.jsonl"
    save_path = "token_to_index.pkl"
    create_dict(train_path, test_path, save_path)