import numpy as np
from transformers import BertTokenizerFast
import torch
from sklearn.model_selection import train_test_split
from utils import *
from models import *
device = 'cuda'
from tqdm import tqdm
tqdm.pandas()


def predict_tl(path, x_test, y_test, t):
    device = 'cuda'
    if t == 'all':
        model = BertUserTwitter(device)
    else:
        model = BertUserTwitterAM(device)
    model.to(device)
    load_checkpoint(path, model)
    model.eval()

    with torch.no_grad():
        # validation loop
        pred, real = [], []
        for tl, label in tqdm(zip(x_test, y_test)):
            if len(tl['tokens']) >= 10:
                atts = split_tensor(tl['tokens'][-get_max_possible_size(len(tl['tokens'])):].to(device), 10)
                masks = split_tensor(tl['mask'][-get_max_possible_size(len(tl['tokens'])):].to(device), 10)
                outputs = model(atts, masks)

                pred.append(outputs.cpu().mean(0).argmax(0).tolist())
                # pred.append(int(outputs.cpu().mean(0)[0] < 0.4))
                real.append(label)
            else:
                pred.append(0)
                real.append(label)

        print(classification_report(real, pred))
        return pred


def predict_tl_final(path, df_test, t, column_text):
    tokenizer = BertTokenizerFast.from_pretrained("pablocosta/bertabaporu-base-uncased")
    # X_test, y_test = get_tokens_mask_dates_split(X, y, tokenizer)
    # dataloader_test = get_dataloader(X_test, y_test)
    device = 'cuda'
    if t == 'all':
        model = BertUserTwitter(device)
    elif t == 'AM':
        model = BertUserTwitterAM(device)
    elif t == 'combiner':
        model = Combiner(device)
    model.to(device)
    load_checkpoint(path, model)
    model.eval()

    with torch.no_grad():
        # validation loop
        pred, real = [], []
        for tl, label, p1, p2 in tqdm(zip(df_test[column_text], df_test['label'], df_test['all_pred'], df_test['AM_pred'])):
            tl = tl[-get_max_possible_size(len(tl)):]
            X_processed = []
            if len(tl) > 0:
                tokenized_texts = tokenizer.batch_encode_plus(tl, max_length=30, padding='max_length',
                                                              truncation=True)
                tokens_tensor = torch.tensor(tokenized_texts['input_ids']).to(device)
                segments_tensors = torch.tensor(tokenized_texts['attention_mask']).to(device)
                # X_processed.append(
                #     {'tokens': tokens_tensor,
                #      'mask': segments_tensors,
                #      }
                # )
            else:
                pred.append(0)
                real.append(label)
                continue
            atts = split_tensor(tokens_tensor, 10)
            masks = split_tensor(segments_tensors, 10)
            outputs = model(atts, masks, torch.Tensor([p1] * len(atts)).to(device), torch.Tensor([p2] * len(atts)).to(device))

            pred.append(outputs.cpu().mean(0).argmax(0).tolist())
            real.append(label)
        print(classification_report(real, pred))
        return pred


def test_models():
    df_train, df_test = load_convert('D', 'all')
    X_train, y_train = get_tokens_mask_dates_split(None, None, None, f'Bert_ALL_train', 'D', '1to10')
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    for i in range(22):
        # path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/ALL_DATA-model-bert_twitter-D-sel-all-epoch{i}_v2.pt'
        path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/ALL_DATA-model-bert_twitter-D-sel-1to10-epoch{i}_v2.pt'
        print(path)
        predict_tl(path, X_val, y_val, 'all')
        print(1)
    """
    D relevancia 28
    D 1to10 22
    D all 38
    """


def run_main_prediction_relevancia_baixo():
    # 1/0
    X_test, y_test = get_tokens_mask_dates_split(None, None, None, f'Bert_relevancia_baixo_test', 'D', 'relevancia')
    for i in [5, 10, 20, 25, 28]:
        path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/ALL_DATA-model-bert_twitter-D-sel-relevancia-baixo-epoch{i}_v2.pt'
        print(path)
        pred_baixo = predict_tl(path, X_test, y_test, 'all')
        print(classification_report(y_test, pred_baixo))


def run_main_prediction_relevancia_baixo_ansiedade():
    # 1/0
    X_test, y_test = get_tokens_mask_dates_split(None, None, None, f'Bert_relevancia_baixo_test', 'A', 'relevancia_baixo')
    for i in [5, 10, 20, 25, 28]:
        path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models//ALL_DATA-model-bert_twitter-A-sel-relevancia_baixo-baixo-epoch{i}_v2.pt'
        print(path)
        pred_baixo = predict_tl(path, X_test, y_test, 'all')
        print(classification_report(y_test, pred_baixo))


def run_main_prediction_relevancia_alto_medio_ansiedade():
    # 1/0
    X_test, y_test = get_tokens_mask_dates_split(None, None, None, f'Bert_relevancia_alto_medio_test', 'A', 'relevancia_alto_medio')
    for i in [5, 7, 10, 15, 17, 20, 22, 25, 28]:
        path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models//ALL_DATA-model-bert_twitter-A-sel-relevancia_alto_medio-baixo-epoch{i}_v2.pt'
        print(path)
        pred_baixo = predict_tl(path, X_test, y_test, 'all')
        print(classification_report(y_test, pred_baixo))


def run_main_prediction():
    X_train, y_train = get_tokens_mask_dates_split(None, None, None, f'Bert_ALL_train', 'D', '1to10')
    X_train, X_val_1to10, y_train, y_val_1to10 = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    X_train, y_train = get_tokens_mask_dates_split(None, None, None, f'Bert_ALL_train', 'D', 'relevancia')
    X_train, X_val_rel, y_train, y_val_rel = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    df_train, df_test = load_convert('D', 'all')
    X_train, X_val_all, y_train, y_val_all = train_test_split(df_train['Text'], df_train['label'], test_size=0.3, random_state=42)
    texts = X_val_all.apply(lambda x: ' # '.join(x)).tolist()
    path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/ALL_DATA-model-bert_twitter-D-sel-1to10-epoch22_v2.pt'
    print(path)
    pred_1to10 = predict_tl(path, X_val_1to10, y_val_1to10, 'all')

    path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/ALL_DATA-model-bert_twitter-D-sel-relevancia-epoch28_v2.pt'
    print(path)
    pred_rel = predict_tl(path, X_val_rel, y_val_rel, 'all')

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    import numpy as np

    # Assuming you have a list 'data' where each element is a tuple (text, flag1, flag2)
    # Replace 'data' with your actual dataset

    # Separate text and flags from the dataset
    texts = [item[0] for item in data]
    flags = np.array([[item[1], item[2]] for item in data])  # Assuming flags are numpy arrays
    texts = [f'p1:{i}\np2:{j}\n{t}' for i, j, t in zip(pred_rel, pred_1to10, texts)]
    # Split the data into training and testing sets
    X_train_text, X_test_text, y_train, y_test = train_test_split(texts, y_val_1to10, test_size=0.2, random_state=42)

    # Convert text data into numerical features using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=10000)  # You can adjust max_features as needed
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_text)
    from sklearn.feature_selection import SelectKBest, chi2
    tfidf_vectorizer = TfidfVectorizer(max_features=10000)  # You can adjust max_features as needed
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

    # Feature selection using chi-square test
    selector = SelectKBest(score_func=chi2, k=500)  # Select top 5000 features
    X_train_selected = selector.fit_transform(X_train_tfidf, y_train)
    X_test_selected = selector.transform(X_test_tfidf)
    class_counts = np.bincount(y_train)
    class_weights = {0: 1.0, 1: class_counts[0] / class_counts[
        1]}  # Weight of class 0 is 1.0, weight of class 1 is proportional to the class imbalance

    # Train a logistic regression classifier
    classifier = LogisticRegression(class_weight=class_weights)
    classifier.fit(X_train_selected, y_train)

    # Predictions
    y_pred = classifier.predict(X_test_selected)

    # Evaluate the classifier
    print(classification_report(y_test, y_pred))


def filter_posts(x, filter):
    posts = []
    for p, l in zip(x['Text'], x['alto_medio_baixo']):
        if l in filter:
            posts.append(p)
    return posts


def get_tokenized(X, y, tokenizer):
    X_processed = []
    for user_posts in tqdm(X):
        # print(tokenizer)
        user_posts = user_posts[-get_max_possible_size(len(user_posts)):]
        if len(user_posts) == 0:
            X_processed.append(
                {'tokens': torch.tensor([]),
                 'mask': torch.tensor([]),
                 }
            )
        else:
            tokenized_texts = tokenizer.batch_encode_plus(user_posts, max_length=30, padding='max_length',
                                                          truncation=True)
            tokens_tensor = torch.tensor(tokenized_texts['input_ids'])
            segments_tensors = torch.tensor(tokenized_texts['attention_mask'])
            X_processed.append(
                {'tokens': tokens_tensor,
                 'mask': segments_tensors,
                 }
            )
    return X_processed, y

def predict_alto_medio_baixo_ansiedade_train_test():
    df_train = pickle.load(open(f'converted_dataframes/A_train_converted.pkl', 'rb'))
    df_test = pickle.load(open(f'converted_dataframes/A_test_converted.pkl', 'rb'))
    # df_train = df_train.sample(n=30)
    df_train['alto_medio'] = df_train.apply(lambda x: filter_posts(x, ['alto', 'medio']), axis=1)
    df_test['alto_medio'] = df_test.apply(lambda x: filter_posts(x, ['alto', 'medio']), axis=1)
    df_train['baixo'] = df_train.apply(lambda x: filter_posts(x, ['baixo']), axis=1)
    df_test['baixo'] = df_test.apply(lambda x: filter_posts(x, ['baixo']), axis=1)

    tokenizer = BertTokenizerFast.from_pretrained("pablocosta/bertabaporu-base-uncased")
    path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models//ALL_DATA-model-bert_twitter-A-sel-relevancia_alto_medio-baixo-epoch23_v2.pt'

    X_test, y_test = get_tokenized(df_train['alto_medio'].tolist(), (df_train['Diagnosed_YN'] == 'yes').astype(int).tolist(), tokenizer)
    pred_alto_medio = predict_tl(path, X_test, y_test, 'all')
    df_train['alto_medio_predicao'] = pred_alto_medio

    X_test, y_test = get_tokenized(df_test['alto_medio'].tolist(), (df_test['Diagnosed_YN'] == 'yes').astype(int).tolist(), tokenizer)
    pred_alto_medio = predict_tl(path, X_test, y_test, 'all')
    df_test['alto_medio_predicao'] = pred_alto_medio

    path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models//ALL_DATA-model-bert_twitter-A-sel-relevancia_baixo-baixo-epoch45_v2.pt'

    X_test, y_test = get_tokenized(df_train['baixo'].tolist(),
                                   (df_train['Diagnosed_YN'] == 'yes').astype(int).tolist(), tokenizer)
    pred_alto_medio = predict_tl(path, X_test, y_test, 'all')
    df_train['baixo_predicao'] = pred_alto_medio

    X_test, y_test = get_tokenized(df_test['baixo'].tolist(),
                                   (df_test['Diagnosed_YN'] == 'yes').astype(int).tolist(), tokenizer)
    pred_alto_medio = predict_tl(path, X_test, y_test, 'all')
    df_test['baixo_predicao'] = pred_alto_medio

    pickle.dump(df_train, open(f'converted_dataframes/A_train_converted_predictions.pkl', 'wb'))
    pickle.dump(df_test, open(f'converted_dataframes/A_test_converted_predictions.pkl', 'wb'))



def get_sample():
    dfa = pickle.load(open(f'converted_dataframes/D_test_converted.pkl', 'rb'))
    dfd = pickle.load(open(f'converted_dataframes/A_test_converted.pkl', 'rb'))
    tweets = []
    for index, row in dfa.iterrows():
        for t, l in zip(row['Text'], row['alto_medio_baixo']):
            tweets.append({
                'id': row['User_ID'],
                "tweet": t,
                "label": l
            })
    for index, row in dfd.iterrows():
        for t, l in zip(row['Text'], row['alto_medio_baixo']):
            tweets.append({
                'id': row['User_ID'],
                "tweet": t,
                "label": l
            })
    random.shuffle(tweets)
    sample = {'alto': [], 'medio': [], 'baixo': []}
    for t in tweets:
        if len(sample[t['label']]) < 200:
            sample[t['label']].append(t)
        if sum([len(sample[s]) for s in sample]) == 600:
            break

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

def normalize(tl):
    tl = eval(tl)
    tl = ' '.join(['C' + str(c) for c in tl])
    return tl


def load_data_mentions(portion, task):
    PATH = '/home/wesley/Documents/doutorado/models/SetembroBR_HAN/data'
    df_c = pd.read_csv(f'{PATH}/{task}/{portion}_{task}_c_SetembroBR_v6.csv', sep=';')
    df_d = pd.read_csv(f'{PATH}/{task}/{portion}_{task}_SetembroBR_v6.csv', sep=';')
    df_c_net = pd.read_csv(f'{PATH}/{task}/{task}_ctrl-users-network_final_v6.csv', sep=';')
    df_d_net = pd.read_csv(f'{PATH}/{task}/{task}_diag-users-network_final_v6.csv', sep=';')
    d_c = dict(zip(df_c_net['User_ID'], df_c_net['Contacts_Anon']))
    d_d = dict(zip(df_d_net['User_ID'], df_d_net['Contacts_Anon']))
    df_d['Text'] = df_d['Text'].apply(lambda x: x.split('#'))
    df_c['Text'] = df_c['Text'].apply(lambda x: x.split('#'))
    df_d['contacts'] = df_d['User_ID'].apply(lambda x: d_d[x])
    df_c['contacts'] = df_c['User_ID'].apply(lambda x: d_c[x])
    return pd.concat((df_c, df_d))

def custom_tokenizer(text):
    # Implement your custom tokenization logic here
    tokens = text.split()  # Example: Split text into tokens by whitespace
    return tokens


def A_prediction_test_mentions():
    df_train = load_data_mentions('train', 'A')
    df_test = load_data_mentions('test', 'A')

    X_train, X_val, y_train, y_val = train_test_split(df_train['contacts'].tolist(),
                                                      (df_train['Diagnosed_YN'] == 'yes').astype(int).tolist(),
                                                      test_size=0.3, random_state=42)
    X_train = [normalize(x) for x in X_train]
    X_val = [normalize(x) for x in X_val]
    X_test = [normalize(x) for x in df_test['contacts'].tolist()]
    y_test = (df_test['Diagnosed_YN'] == 'yes').astype(int).tolist()

    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)

    # Convert class weights to dictionary format
    class_weights_dict = dict(zip(classes, class_weights))

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=custom_tokenizer)),
        ('anova', SelectKBest(f_classif, k=20_000)),
        # ('clf', Lasso(alpha=0.1, class_weight=class_weights_dict)),  # Elastic Net regularization
        ('clf', LogisticRegression(class_weight=class_weights_dict, penalty='l2', C=1.0))
        # ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))
        # Gradient Boosting classifier with class weights        # Use 'balanced' to automatically adjust weights
    ])

    # Step 4: Fit the pipeline
    pipeline.fit(X_train, y_train)

    # Step 5: Evaluate the model
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    return y_pred
    # y_pred = pipeline.predict(X_test)
    # print(classification_report(y_test, y_pred))
    print(1)

def D_prediction_test_mentions():
    df_train = load_data_mentions('train', 'D')
    df_test = load_data_mentions('test', 'D')

    X_train, X_val, y_train, y_val = train_test_split(df_train['contacts'].tolist(), (df_train['Diagnosed_YN'] == 'yes').astype(int).tolist(), test_size=0.3, random_state=42)
    X_train = [normalize(x) for x in X_train]
    X_val = [normalize(x) for x in X_val]
    X_test = [normalize(x) for x in df_test['contacts'].tolist()]
    y_test = (df_test['Diagnosed_YN'] == 'yes').astype(int).tolist()

    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)

    # Convert class weights to dictionary format
    class_weights_dict = dict(zip(classes, class_weights))

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=custom_tokenizer)),
        ('anova', SelectKBest(f_classif, k=20_000)),
        # ('clf', Lasso(alpha=0.1, class_weight=class_weights_dict)),  # Elastic Net regularization
        ('clf', LogisticRegression(class_weight=class_weights_dict, penalty='l2', C=1.0))
        # ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))
        # Gradient Boosting classifier with class weights        # Use 'balanced' to automatically adjust weights
    ])

    # Step 4: Fit the pipeline
    pipeline.fit(X_train, y_train)

    # Step 5: Evaluate the model
    y_pred = pipeline.predict(X_val)
    print(classification_report(y_val, y_pred))
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    return y_pred

def A_prediction_train_mentions():
    df_train = load_data_mentions('train', 'A')
    df_test = load_data_mentions('test', 'A')

    X_train, X_val, y_train, y_val = train_test_split(df_train['contacts'].tolist(),
                                                      (df_train['Diagnosed_YN'] == 'yes').astype(int).tolist(),
                                                      test_size=0.3, random_state=42)
    X_train = [normalize(x) for x in X_train]
    X_val = [normalize(x) for x in X_val]
    X_test = [normalize(x) for x in df_test['contacts'].tolist()]
    y_test = (df_test['Diagnosed_YN'] == 'yes').astype(int).tolist()

    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)

    # Convert class weights to dictionary format
    class_weights_dict = dict(zip(classes, class_weights))

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=custom_tokenizer)),
        ('anova', SelectKBest(f_classif, k=20_000)),
        # ('clf', Lasso(alpha=0.1, class_weight=class_weights_dict)),  # Elastic Net regularization
        ('clf', LogisticRegression(class_weight=class_weights_dict, penalty='l2', C=1.0))
        # ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))
        # Gradient Boosting classifier with class weights        # Use 'balanced' to automatically adjust weights
    ])

    # Step 4: Fit the pipeline
    pipeline.fit(X_train, y_train)

    # Step 5: Evaluate the model
    y_pred = pipeline.predict(X_val)
    print(classification_report(y_val, y_pred))
    return y_pred
    # y_pred = pipeline.predict(X_test)
    # print(classification_report(y_test, y_pred))
    print(1)

def D_prediction_train_mentions():
    df_train = load_data_mentions('train', 'D')
    df_test = load_data_mentions('test', 'D')

    X_train, X_val, y_train, y_val = train_test_split(df_train['contacts'].tolist(), (df_train['Diagnosed_YN'] == 'yes').astype(int).tolist(), test_size=0.3, random_state=42)
    X_train = [normalize(x) for x in X_train]
    X_val = [normalize(x) for x in X_val]
    X_test = [normalize(x) for x in df_test['contacts'].tolist()]
    y_test = (df_test['Diagnosed_YN'] == 'yes').astype(int).tolist()

    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)

    # Convert class weights to dictionary format
    class_weights_dict = dict(zip(classes, class_weights))

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=custom_tokenizer)),
        ('anova', SelectKBest(f_classif, k=20_000)),
        # ('clf', Lasso(alpha=0.1, class_weight=class_weights_dict)),  # Elastic Net regularization
        ('clf', LogisticRegression(class_weight=class_weights_dict, penalty='l2', C=1.0))
        # ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))
        # Gradient Boosting classifier with class weights        # Use 'balanced' to automatically adjust weights
    ])

    # Step 4: Fit the pipeline
    pipeline.fit(X_train, y_train)

    # Step 5: Evaluate the model
    y_pred = pipeline.predict(X_val)
    print(classification_report(y_val, y_pred))
    y_pred = pipeline.predict(X_val)
    print(classification_report(y_val, y_pred))
    return y_pred


def predict_bert_depressao_train_test():
    X_test, y_test = get_tokens_mask_dates_split(None, None, None, f'Bert_LSTM_train', "D", "all")
    _, X_test, _, y_test = train_test_split(X_test,y_test, test_size=0.3, random_state=42)
    pred_ment = D_prediction_train_mentions()

    tokenizer = BertTokenizerFast.from_pretrained("pablocosta/bertabaporu-base-uncased")
    for i in range(10, 20):
        i = 16
        path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models//ALL_DATA-model-bert_twitter-D-sel-all-baixo-epoch{i}_v2.pt'

        # path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models//ALL_DATA-model-bert_twitter-A-sel-relevancia_baixo-baixo-epoch45_v2.pt'

        pred = predict_tl(path, X_test, y_test, 'all')
        pickle.dump([pred_ment, pred, y_test], open("D_train_bert_predictions.pkl", "wb"))
        break


def predict_bert_ansiedade_train_test():
    X_test, y_test = get_tokens_mask_dates_split(None, None, None, f'Bert_LSTM_train', "A", "all")
    _, X_test, _, y_test = train_test_split(X_test,y_test, test_size=0.3, random_state=42)

    pred_ment = A_prediction_train_mentions()
    tokenizer = BertTokenizerFast.from_pretrained("pablocosta/bertabaporu-base-uncased")
    for i in range(5, 20):
        i = 8
        path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models//ALL_DATA-model-bert_twitter-A-sel-all-baixo-epoch{i}_v2.pt'

        # path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models//ALL_DATA-model-bert_twitter-A-sel-relevancia_baixo-baixo-epoch45_v2.pt'

        pred = predict_tl(path, X_test, y_test, 'all')
        pickle.dump([pred_ment, pred, y_test], open("A_train_bert_predictions.pkl", "wb"))
        break


def predict_bert_depressao_test():
    X_test, y_test = get_tokens_mask_dates_split(None, None, None, f'Bert_LSTM_test', "D", "all")
    pred_ment = D_prediction_test_mentions()

    tokenizer = BertTokenizerFast.from_pretrained("pablocosta/bertabaporu-base-uncased")
    for i in range(10, 20):
        i = 16
        path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models//ALL_DATA-model-bert_twitter-D-sel-all-baixo-epoch{i}_v2.pt'

        # path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models//ALL_DATA-model-bert_twitter-A-sel-relevancia_baixo-baixo-epoch45_v2.pt'

        pred = predict_tl(path, X_test, y_test, 'all')
        pickle.dump([pred_ment, pred, y_test], open("D_test_bert_predictions.pkl", "wb"))
        break


def predict_bert_ansiedade_test():
    X_test, y_test = get_tokens_mask_dates_split(None, None, None, f'Bert_LSTM_test', "A", "all")

    pred_ment = A_prediction_train_mentions()
    tokenizer = BertTokenizerFast.from_pretrained("pablocosta/bertabaporu-base-uncased")
    for i in range(5, 20):
        i = 8
        path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models//ALL_DATA-model-bert_twitter-A-sel-all-baixo-epoch{i}_v2.pt'

        # path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models//ALL_DATA-model-bert_twitter-A-sel-relevancia_baixo-baixo-epoch45_v2.pt'

        pred = predict_tl(path, X_test, y_test, 'all')
        pickle.dump([pred_ment, pred, y_test], open("A_test_bert_predictions.pkl", "wb"))
        break


def read_predictions():
    pickle.dump(open("D_test_bert_predictions.pkl", "rb"))

# get_sample()

if __name__ == '__main__':
    # predict_bert_depressao_train_test()
    # predict_bert_ansiedade_train_test()
    # predict_bert_depressao_test()
    # predict_bert_ansiedade_test()
    # predict_alto_medio_baixo_ansiedade_train_test()
    run_main_prediction_relevancia_alto_medio_ansiedade()
    run_main_prediction_relevancia_baixo_ansiedade()
    # ALL_DATA-model-bert_twitter-A-sel-relevancia_baixo-baixo-epoch25_v2.pt
    # ALL_DATA-model-bert_twitter-A-sel-relevancia_alto_medio-baixo-epoch25_v2.pt