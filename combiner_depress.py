import pandas as pd
import pandas
from models import *
from utils import *
from tester_depress import *
from sklearn.model_selection import train_test_split
from tester_depress import predict_tl
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

device = 'cuda'


def get_data():
    return pickle.load(open('Data_D_validation_to_train.pkl', 'rb'))

    df_train, _ = load_convert('D', 'all')
    df = pickle.load(open(f'/home/wesley/Documents/projects/llm_experiments/sbr/treino_D_relevancia.pkl', 'rb'))
    dict_predictions = dict(zip(df['User_ID'], df['tl_prediction']))


    X_train, X_val, y_train, y_val = train_test_split(df_train[["User_ID", 'Text']], df_train['label'], test_size=0.2, random_state=42)
    X_val['label'] = y_val
    # df = pd.DataFrame({"Text": X_val, 'label': y_val}).sample(n=20)
    path = '/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/model-bert_twitter-D-sel-all_br-epoch29.pt'
    # X_val = X_val.sample(n=100)
    # X_val['all_pred'] = predict_tl(path, X_val, 'all', 'Text')

    labels = ['alto']
    # X_val = X_val.sample(n=500)
    X_val['clean_text'] = X_val.apply(lambda x: [x1 for x1, x2 in zip(x['Text'], dict_predictions[x['User_ID']]) if x2 in labels], axis=1)
    for i in range(2, 16, 1):
        path = f'/home/wesley/Documents/projects/llm_experiments/sbr/second_level_models/model_second_level_bert_{i}_a.pt'
        print(path)
        try:
            X_val['AM_pred'] = predict_tl(path, X_val, 'AM', 'clean_text')
        except:
            continue
    exit()
    pickle.dump([X_val, dict_predictions], open('Data_D_validation_to_train.pkl', 'wb'))
    return X_val


def train(task, sel, type_):
    X_val, dict_predictions = get_data()
    tokenizer = BertTokenizerFast.from_pretrained("pablocosta/bertweet-br-base-uncased")


    X_train, y_train = get_tokens_mask_dates_split(X_val['Text'].to_numpy(), X_val['label'].to_numpy(),
                                                   tokenizer, f'{type_}_train_with_eval', task, sel)
    # return X_train, y_train
    pred_1 = X_val['all_pred'].to_numpy()
    pred_2 = X_val['AM_pred'].to_numpy()
    dataloader_train = get_dataloader_combiner(X_train, y_train, pred_1, pred_2, False)

    model = Combiner(device)
    model.to(device)
    epoch_wait = 0
    # file_path = f'/content/drive/MyDrive/doutorado/diag_trat_saving/{sel}/ALL_DATA-model-{model_name}-{task}-sel-{sel}-epoch{epoch_wait}.pt'
    # print('*' * 20)
    # load_checkpoint(file_path, model)

    lr = 1e-5
    optimizer = AdamW(model.parameters(), lr=lr)

    train_loss_set = []
    epochs = 150
    file_path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/'
    vect_weights = create_weight_dict(y_train)
    print(vect_weights)
    loss_fct = CrossEntropyLoss(weight=torch.tensor(vect_weights, dtype=torch.float32), reduction='mean')
    loss_fct.to(device)
    for epoch in range(epochs):
        # if epoch_wait > epoch:
        #     continue
        print(f'Epoch {epoch}')
        model.train()
        pred, true_label = [], []
        print(len(dataloader_train))
        for i, batch in enumerate(dataloader_train):
            if i % 30 == 0:
                # break
                print(datetime.fromtimestamp(time.time()), i, train_loss_set[-1] if len(train_loss_set) else 0)
            # continue
            atts = torch.stack([aa[0]['tokens'] for aa in batch]).to(device)
            masks = torch.stack([aa[0]['mask'] for aa in batch]).to(device)
            p1 = torch.Tensor([b[1] for b in batch]).to(device)
            p2 = torch.Tensor([b[2] for b in batch]).to(device)
            outputs = model(atts, masks, p1, p2)
            loss = loss_fct(outputs, torch.tensor([b[3] for b in batch]).to(device))
            # output = loss(torch.argmax(outputs[1], dim=1).add_(-1), b_labels.add_(-1))
            train_loss_set.append(loss.item())
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()

            pred = pred + outputs.cpu().argmax(1).tolist()
            true_label = true_label + [b[1] for b in batch]
            gc.collect()
            torch.cuda.empty_cache()
        # continue
        print('Results')
        print(classification_report(true_label, pred))
        save_checkpoint(file_path + '/' + f'COMBINER-model-{task}-sel-{sel}-epoch{epoch}.pt',
                        model, 0)


def get_test_data():
    # return pickle.load(open('teste_data_combination.pkl', 'rb'))
    _, df_test = load_convert('D', 'all')
    df = pickle.load(open(f'/home/wesley/Documents/projects/llm_experiments/sbr/teste_D_relevancia.pkl', 'rb'))
    dict_predictions = dict(zip(df['User_ID'], df['label']))
    labels = ['alto']
    df_test['clean_text'] = df_test.apply(
        lambda x: [x1 for x1, x2 in zip(x['Text'], dict_predictions[x['User_ID']]) if x2 in labels], axis=1)
    # for i in range(20):
    path = f'/home/wesley/Documents/projects/llm_experiments/sbr/second_level_models/model_second_level_bert_2_a.pt'
    df_test['AM_pred'] = predict_tl(path, df_test, 'AM', 'clean_text')
    path = '/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/model-bert_twitter-D-sel-all_br-epoch29.pt'
    df_test['all_pred'] = predict_tl(path, df_test, 'all', 'Text')

    pickle.dump([df_test, dict_predictions], open('teste_data_combination.pkl', 'wb'))
    return df_test, dict_predictions


def test_data():
    df, _ = get_test_data()
    path = '/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/COMBINER-model-D-sel-all-epoch5.pt'
    predict_tl_final(path, df, 'combiner', 'Text')
    print(1)

# test_data()
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


def prediction_val_mentions():
    df_train = load_data_mentions('train', 'D')
    # df_test = load_data_mentions('test', 'D')

    X_train, X_val, y_train, y_val = train_test_split(df_train['contacts'].tolist(), (df_train['Diagnosed_YN'] == 'yes').astype(int).tolist(), test_size=0.3, random_state=42)
    X_train = [normalize(x) for x in X_train]
    X_val = [normalize(x) for x in X_val]
    # X_test = [normalize(x) for x in df_test['contacts'].tolist()]
    # y_test = (df_test['Diagnosed_YN'] == 'yes').astype(int).tolist()

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


def prediction_test_mentions():
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


def run_main_prediction():
    from sklearn.model_selection import train_test_split
    X_train, y_train = get_tokens_mask_dates_split(None, None, None, f'Bert_ALL_train', 'D', '1to10')
    X_train, X_val_1to10, y_train, y_val_1to10 = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    X_train, y_train = get_tokens_mask_dates_split(None, None, None, f'Bert_ALL_train', 'D', 'relevancia')
    X_train, X_val_rel, y_train, y_val_rel = train_test_split(X_train, y_train, test_size=0.3, random_state=42)


    df_train, df_test = load_convert('D', 'all')
    X_train, X_val_all, y_train, y_val_all = train_test_split(df_train['Text'], df_train['label'], test_size=0.3, random_state=42)
    texts = X_val_all.apply(lambda x: ' # '.join(x)).tolist()
    path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/ALL_DATA-model-bert_twitter-D-sel-1to10-epoch22_v2.pt'
    print(path)
    pred_mentions = prediction_val_mentions()
    pred_1to10 = predict_tl(path, X_val_1to10, y_val_1to10, 'all')

    path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/ALL_DATA-model-bert_twitter-D-sel-relevancia-epoch28_v2.pt'
    print(path)
    pred_rel = predict_tl(path, X_val_rel, y_val_rel, 'all')
    tokenizer = BertTokenizerFast.from_pretrained("pablocosta/bertweet-br-base-uncased")
    X_tok, y_tok = get_tokens_mask_dates_split(X_val_all.tolist(), y_val_all.tolist(),
                                                   tokenizer, f'combiner_train', 'D', 'all')

    dataloader_train = get_dataloader_combiner(X_tok, y_tok, pred_rel, pred_1to10, pred_mentions, False)

    model = Combiner(device)
    model.to(device)
    epoch_wait = 0
    # file_path = f'/content/drive/MyDrive/doutorado/diag_trat_saving/{sel}/ALL_DATA-model-{model_name}-{task}-sel-{sel}-epoch{epoch_wait}.pt'
    # print('*' * 20)
    # load_checkpoint(file_path, model)

    lr = 1e-5
    optimizer = AdamW(model.parameters(), lr=lr)

    train_loss_set = []
    epochs = 150
    file_path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/'
    vect_weights = create_weight_dict(y_train)
    print(vect_weights)
    loss_fct = CrossEntropyLoss(weight=torch.tensor(vect_weights, dtype=torch.float32), reduction='mean')
    loss_fct.to(device)
    for epoch in range(epochs):
        # if epoch_wait > epoch:
        #     continue
        print(f'Epoch {epoch}')
        model.train()
        pred, true_label = [], []
        print(len(dataloader_train))
        for i, batch in enumerate(dataloader_train):
            if i % 30 == 0:
                # break
                print(datetime.fromtimestamp(time.time()), i, train_loss_set[-1] if len(train_loss_set) else 0)
            # continue
            atts = torch.stack([aa[0]['tokens'] for aa in batch]).to(device)
            masks = torch.stack([aa[0]['mask'] for aa in batch]).to(device)
            p1 = torch.Tensor([b[1] for b in batch]).to(device)
            p2 = torch.Tensor([b[2] for b in batch]).to(device)
            p3 = torch.Tensor([b[3] for b in batch]).to(device)
            outputs = model(atts, masks, p1, p2, p3)
            loss = loss_fct(outputs, torch.tensor([b[3] for b in batch]).to(device))
            # output = loss(torch.argmax(outputs[1], dim=1).add_(-1), b_labels.add_(-1))
            train_loss_set.append(loss.item())
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()

            pred = pred + outputs.cpu().argmax(1).tolist()
            true_label = true_label + [b[1] for b in batch]
            gc.collect()
            torch.cuda.empty_cache()
        # continue
        print('Results')
        print(classification_report(true_label, pred))
        save_checkpoint(file_path + '/' + f'COMBINER-model-D-sel-combiner-epoch{epoch}.pt',
                        model, 0)


def run_main_prediction_test():

    try:
        # 1/0
        X_test_all, y_test_all, pred_rel, pred_rel_baixo, pred_1to10, pred_mentions = pickle.load(open('combiner_test_dl.pkl', 'rb'))
    except:
        from sklearn.model_selection import train_test_split
        X_test_rel_baixo, y_test_rel_baixo = get_tokens_mask_dates_split(None, None, None, f'Bert_relevancia_baixo_test', 'D', 'relevancia')
        X_test_1to10, y_test_1to10 = get_tokens_mask_dates_split(None, None, None, f'Bert_ALL_test', 'D', '1to10')
        X_test_relevancia, y_test_relevancia = get_tokens_mask_dates_split(None, None, None, f'Bert_ALL_test', 'D', 'relevancia')
        X_test_all, y_test_all = get_tokens_mask_dates_split(None, None, None, f'Bert_ALL_test', 'D', 'all')

        pred_mentions = prediction_test_mentions()

        path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/ALL_DATA-model-bert_twitter-D-sel-relevancia-baixo-epoch50_v2.pt'
        print(path)
        pred_rel_baixo = predict_tl(path, X_test_rel_baixo, y_test_rel_baixo, 'all')


        path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/ALL_DATA-model-bert_twitter-D-sel-1to10-epoch22_v2.pt'
        print(path)
        pred_1to10 = predict_tl(path, X_test_1to10, y_test_1to10, 'all')

        path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/ALL_DATA-model-bert_twitter-D-sel-relevancia-epoch28_v2.pt'
        print(path)
        pred_rel = predict_tl(path, X_test_relevancia, y_test_relevancia, 'all')
        pickle.dump([X_test_all, y_test_all, pred_rel, pred_rel_baixo, pred_1to10, pred_mentions], open('combiner_test_dl.pkl', 'wb'))
    model = Combiner(device)
    model.to(device)

    file_path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/'
    for i in range(10):
        path = file_path + '/' + f'COMBINER-model-D-sel-combiner-epoch{i}.pt'
        load_checkpoint(path, model)
        with torch.no_grad():
            # validation loop
            pred, real = [], []
            for tl, label, p1, p2, p3 in tqdm(zip(X_test_all, y_test_all, pred_rel, pred_1to10, pred_mentions)):
                if len(tl['tokens']) >= 10:
                    atts = split_tensor(tl['tokens'][-get_max_possible_size(len(tl['tokens'])):].to(device), 10)
                    masks = split_tensor(tl['mask'][-get_max_possible_size(len(tl['tokens'])):].to(device), 10)
                    outputs = model(atts, masks, [p1]*len(atts), [p2]*len(atts), [p3]*len(atts))

                    pred.append(outputs.cpu().mean(0).argmax(0).tolist())
                    # pred.append(int(outputs.cpu().mean(0)[0] < 0.4))
                    real.append(label)
                else:
                    pred.append(0)
                    real.append(label)

            print(classification_report(real, pred))
            # return pred



def run_main_prediction_reglog():
    from sklearn.model_selection import train_test_split
    X_train, y_train = get_tokens_mask_dates_split(None, None, None, f'Bert_ALL_train', 'D', '1to10')
    X_train, X_val_1to10, y_train, y_val_1to10 = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    X_train, y_train = get_tokens_mask_dates_split(None, None, None, f'Bert_ALL_train', 'D', 'relevancia')
    X_train, X_val_rel, y_train, y_val_rel = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    df_train, df_test = load_convert('D', 'all')
    X_train, X_val_all, y_train, y_val_all = train_test_split(df_train['Text'], df_train['label'], test_size=0.3, random_state=42)
    texts = X_val_all.apply(lambda x: ' # '.join(x)).tolist()
    path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/ALL_DATA-model-bert_twitter-D-sel-1to10-epoch22_v2.pt'
    print(path)
    pred_mentions = prediction_val_mentions()
    pred_1to10 = predict_tl(path, X_val_1to10, y_val_1to10, 'all')


    pred_rel = predict_tl(path, X_val_rel, y_val_rel, 'all')
    X_train = texts[:2000]
    X_test = texts[2000:]
    y_train = y_val_all.tolist()[:2000]
    y_test = y_val_all.tolist()[2000:]
    tokenizer = TfidfVectorizer(tokenizer=custom_tokenizer)
    X_train_tok = tokenizer.fit_transform(X_train)
    X_eval_tok = tokenizer.transform(X_test)

    sel = SelectKBest(f_classif, k=20_000)
    X_train_selected = sel.fit_transform(X_train_tok, y_train).toarray()
    X_val_selected = sel.transform(X_eval_tok).toarray()
    X_train_concat = np.concatenate(
        [X_train_selected, np.array(pred_rel[:2000]).reshape(-1, 1), np.array(pred_1to10[:2000]).reshape(-1, 1),
         pred_mentions[:2000].reshape(-1, 1)], axis=1)
    X_test_concat = np.concatenate(
        [X_val_selected, np.array(pred_rel[2000:]).reshape(-1, 1), np.array(pred_1to10[2000:]).reshape(-1, 1),
         pred_mentions[2000:].reshape(-1, 1)], axis=1)
    # Step 4: Fit the pipeline
    # pipeline.fit(X_train, y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y_train)
    class_weights_dict = dict(zip([0, 1], class_weights))

    logreg = LogisticRegression(class_weight=class_weights_dict, penalty='l2', C=1.0)

    # Step 5: Evaluate the model
    logreg.fit(X_train_concat, y_train)
    y_pred = logreg.predict(X_train_selected)
    print(classification_report(y_train, y_pred))
    y_pred = logreg.predict(X_val_selected)
    print(classification_report(y_test, y_pred))
    print(1)


def run_main_prediction_on_test():
    from sklearn.model_selection import train_test_split
    X_train, y_train = get_tokens_mask_dates_split(None, None, None, f'Bert_ALL_train', 'D', '1to10')
    X_train, X_val_1to10, y_train, y_val_1to10 = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    X_train, y_train = get_tokens_mask_dates_split(None, None, None, f'Bert_ALL_train', 'D', 'relevancia')
    X_train, X_val_rel, y_train, y_val_rel = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    df_train, df_test = load_convert('D', 'all')
    X_train, X_val_all, y_train, y_val_all = train_test_split(df_train['Text'], df_train['label'], test_size=0.3, random_state=42)
    texts = X_val_all.apply(lambda x: ' # '.join(x)).tolist()
    path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/ALL_DATA-model-bert_twitter-D-sel-1to10-epoch22_v2.pt'
    print(path)
    pred_mentions = prediction_val_mentions()
    pred_1to10 = predict_tl(path, X_val_1to10, y_val_1to10, 'all')

    path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/ALL_DATA-model-bert_twitter-D-sel-relevancia-epoch28_v2.pt'
    print(path)
    pred_rel = predict_tl(path, X_val_rel, y_val_rel, 'all')
    tokenizer = BertTokenizerFast.from_pretrained("pablocosta/bertweet-br-base-uncased")
    X_tok, y_tok = get_tokens_mask_dates_split(X_val_all.tolist(), y_val_all.tolist(),
                                                   tokenizer, f'combiner_train', 'D', 'all')

    dataloader_train = get_dataloader_combiner(X_tok, y_tok, pred_rel, pred_1to10, pred_mentions, False)

    model = Combiner(device)
    model.to(device)
    epoch_wait = 0
    # file_path = f'/content/drive/MyDrive/doutorado/diag_trat_saving/{sel}/ALL_DATA-model-{model_name}-{task}-sel-{sel}-epoch{epoch_wait}.pt'
    # print('*' * 20)
    # load_checkpoint(file_path, model)

    lr = 1e-5
    optimizer = AdamW(model.parameters(), lr=lr)

    train_loss_set = []
    epochs = 150
    file_path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/'
    vect_weights = create_weight_dict(y_train)
    print(vect_weights)
    loss_fct = CrossEntropyLoss(weight=torch.tensor(vect_weights, dtype=torch.float32), reduction='mean')
    loss_fct.to(device)
    for epoch in range(epochs):
        # if epoch_wait > epoch:
        #     continue
        print(f'Epoch {epoch}')
        model.train()
        pred, true_label = [], []
        print(len(dataloader_train))
        for i, batch in enumerate(dataloader_train):
            if i % 30 == 0:
                # break
                print(datetime.fromtimestamp(time.time()), i, train_loss_set[-1] if len(train_loss_set) else 0)
            # continue
            atts = torch.stack([aa[0]['tokens'] for aa in batch]).to(device)
            masks = torch.stack([aa[0]['mask'] for aa in batch]).to(device)
            p1 = torch.Tensor([b[1] for b in batch]).to(device)
            p2 = torch.Tensor([b[2] for b in batch]).to(device)
            p3 = torch.Tensor([b[3] for b in batch]).to(device)
            outputs = model(atts, masks, p1, p2, p3)
            loss = loss_fct(outputs, torch.tensor([b[3] for b in batch]).to(device))
            # output = loss(torch.argmax(outputs[1], dim=1).add_(-1), b_labels.add_(-1))
            train_loss_set.append(loss.item())
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()

            pred = pred + outputs.cpu().argmax(1).tolist()
            true_label = true_label + [b[1] for b in batch]
            gc.collect()
            torch.cuda.empty_cache()
        # continue
        print('Results')
        print(classification_report(true_label, pred))
        save_checkpoint(file_path + '/' + f'COMBINER-model-D-sel-combiner-epoch{epoch}.pt',
                        model, 0)


def train_with_validation_combiner():
    try:
        # 1/0
        X_text, y_real_label, pred_rel, pred_rel_baixo, pred_1to10, pred_mentions = pickle.load(open('combiner_val_dl.pkl', 'rb'))
    except:
        from sklearn.model_selection import train_test_split
        # 1/0
        X_train, y_train = get_tokens_mask_dates_split(None, None, None, f'Bert_relevancia_baixo_train', 'D', 'relevancia')
        X_test, y_test = get_tokens_mask_dates_split(None, None, None, f'Bert_relevancia_baixo_test', 'D', 'relevancia')
        X_train, X_val_rel_baixo, y_train, y_val_rel_baixo = train_test_split(X_train, y_train, test_size=0.3, random_state=42)


        X_train_1to10, y_train_1to10 = get_tokens_mask_dates_split(None, None, None, f'Bert_ALL_train', 'D', '1to10')
        X_train, X_val_1to10, y_train, y_val_1to10 = train_test_split(X_train_1to10, y_train_1to10, test_size=0.3,
                                                                              random_state=42)

        X_train_rel_am, y_train_rel_am = get_tokens_mask_dates_split(None, None, None, f'Bert_ALL_train', 'D', 'relevancia')
        X_train, X_val_rel_am, y_train, y_val_rel_am = train_test_split(X_train_rel_am, y_train_rel_am, test_size=0.3,
                                                                      random_state=42)

        pred_mentions = prediction_val_mentions()

        path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/ALL_DATA-model-bert_twitter-D-sel-relevancia-baixo-epoch50_v2.pt'
        print(path)
        pred_rel_baixo = predict_tl(path, X_val_rel_baixo, y_val_rel_baixo, 'all')


        path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/ALL_DATA-model-bert_twitter-D-sel-1to10-epoch22_v2.pt'
        print(path)
        pred_1to10 = predict_tl(path, X_val_1to10, y_val_1to10, 'all')

        path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/ALL_DATA-model-bert_twitter-D-sel-relevancia-epoch28_v2.pt'
        print(path)
        pred_rel = predict_tl(path, X_val_rel_am, y_val_rel_am, 'all')

        df_train, _ = load_convert('D', 'all')
        _, X_text, _, y_real_label = train_test_split(df_train['Text'].tolist(), df_train['label'].tolist(), test_size=0.3,
                                                                      random_state=42)
        pickle.dump([X_text, y_real_label, pred_rel, pred_rel_baixo, pred_1to10, pred_mentions], open('combiner_val_dl.pkl', 'wb'))

    print(1)
    x_input = np.stack((np.array(pred_rel), np.array(pred_rel_baixo), np.array(pred_1to10), pred_mentions)).T
    texts = ['#'.join(x) for x in X_text]
    xtrain = x_input[:2000]
    ytrain = y_real_label[:2000]
    xtest = x_input[2000:]
    ytest = y_real_label[2000:]

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    embeddings = model.encode(texts)
    text_train = embeddings[:2000]
    text_test = embeddings[2000:]

    dataloader_train = get_dataloader_combiner(text_train, ytrain, xtrain, False)
    dataloader_test = get_dataloader_combiner(text_test, ytest, xtest, False)


    model = CombinedModel(input_size=768, num_models=4)
    # load_checkpoint('/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/COMBINER-model-D-gating-sel-combiner-epoch149.pt', model)
    model.to(device)
    lr = 1e-6
    optimizer = AdamW(model.parameters(), lr=lr)

    train_loss_set = []
    epochs = 150
    file_path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/'
    vect_weights = create_weight_dict(ytrain)
    print(vect_weights)
    loss_fct = CrossEntropyLoss(weight=torch.tensor(vect_weights, dtype=torch.float32), reduction='mean')
    loss_fct.to(device)
    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        model.train()
        pred, true_label = [], []
        for i, batch in enumerate(dataloader_train):
            if i % 30 == 0:
                # break
                print(datetime.fromtimestamp(time.time()), i, train_loss_set[-1] if len(train_loss_set) else 0)
            # continue
            embs = torch.stack([torch.Tensor(aa[0]) for aa in batch]).to(device)
            predictions = torch.stack([torch.Tensor(aa[1]) for aa in batch]).to(device)
            outputs = model(embs, predictions)
            loss = loss_fct(outputs, torch.tensor([aa[2] for aa in batch]).to(device))
            # output = loss(torch.argmax(outputs[1], dim=1).add_(-1), b_labels.add_(-1))
            train_loss_set.append(loss.item())
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()

            pred = pred + outputs.cpu().argmax(1).tolist()
            true_label = true_label + [b[2] for b in batch]
            gc.collect()
            torch.cuda.empty_cache()
        # continue
        print('Results')
        print(classification_report(true_label, pred))
        save_checkpoint(file_path + '/' + f'COMBINER-model-D-gating-sel-combiner-epoch{epoch}.pt',
                        model, 0)
        with torch.no_grad():
            # validation loop
            model.eval()
            pred, real = [], []
            for i, batch in enumerate(dataloader_test):
                embs = torch.stack([torch.Tensor(aa[0]) for aa in batch]).to(device)
                predictions = torch.stack([torch.Tensor(aa[1]) for aa in batch]).to(device)
                outputs = model(embs, predictions)
                pred = pred + outputs.cpu().argmax(1).tolist()
                real = real + [b[2] for b in batch]
            print(classification_report(real, pred))


def train_with_full_timeline_combiner(rel_al, rel_b, ment, s1to10_5):
    try:
        # 1/0
        X_text, y_real_label, pred_rel, pred_rel_baixo, pred_1to10, pred_mentions = pickle.load(open('combiner_val_dl.pkl', 'rb'))
    except:
        from sklearn.model_selection import train_test_split
        # 1/0
        X_train, y_train = get_tokens_mask_dates_split(None, None, None, f'Bert_relevancia_baixo_train', 'D', 'relevancia')
        X_test, y_test = get_tokens_mask_dates_split(None, None, None, f'Bert_relevancia_baixo_test', 'D', 'relevancia')
        X_train, X_val_rel_baixo, y_train, y_val_rel_baixo = train_test_split(X_train, y_train, test_size=0.3, random_state=42)


        X_train_1to10, y_train_1to10 = get_tokens_mask_dates_split(None, None, None, f'Bert_ALL_train', 'D', '1to10')
        X_train, X_val_1to10, y_train, y_val_1to10 = train_test_split(X_train_1to10, y_train_1to10, test_size=0.3,
                                                                              random_state=42)

        X_train_rel_am, y_train_rel_am = get_tokens_mask_dates_split(None, None, None, f'Bert_ALL_train', 'D', 'relevancia')
        X_train, X_val_rel_am, y_train, y_val_rel_am = train_test_split(X_train_rel_am, y_train_rel_am, test_size=0.3,
                                                                      random_state=42)

        pred_mentions = prediction_val_mentions()

        path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/ALL_DATA-model-bert_twitter-D-sel-relevancia-baixo-epoch50_v2.pt'
        print(path)
        pred_rel_baixo = predict_tl(path, X_val_rel_baixo, y_val_rel_baixo, 'all')


        path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/ALL_DATA-model-bert_twitter-D-sel-1to10-epoch22_v2.pt'
        print(path)
        pred_1to10 = predict_tl(path, X_val_1to10, y_val_1to10, 'all')

        path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/ALL_DATA-model-bert_twitter-D-sel-relevancia-epoch28_v2.pt'
        print(path)
        pred_rel = predict_tl(path, X_val_rel_am, y_val_rel_am, 'all')

        df_train, _ = load_convert('D', 'all')
        _, X_text, _, y_real_label = train_test_split(df_train['Text'].tolist(), df_train['label'].tolist(), test_size=0.3,
                                                                      random_state=42)
        pickle.dump([X_text, y_real_label, pred_rel, pred_rel_baixo, pred_1to10, pred_mentions], open('combiner_val_dl.pkl', 'wb'))

    print(1)
    models = []
    if rel_al:
        models.append(np.array(pred_rel))
    if rel_b:
        models.append(np.array(pred_rel_baixo))
    if ment:
        models.append(np.array(pred_1to10))
    if s1to10_5:
        models.append(pred_mentions)
    models.append(np.ones(len(pred_rel_baixo)))

    x_input = np.stack(models).T
    texts = ['#'.join(reversed(x)) for x in X_text]
    X_processed = []
    tokenizer = BertTokenizerFast.from_pretrained("pablocosta/bertweet-br-base-uncased")
    tokenized_texts = tokenizer.batch_encode_plus(texts, max_length=512, padding='max_length',
                                                              truncation=True)
    tokens_tensor = [torch.tensor(t) for t in tokenized_texts['input_ids']]
    segments_tensors = [torch.tensor(t) for t in tokenized_texts['attention_mask']]

    xtrain = x_input[:2000]
    ytrain = y_real_label[:2000]
    xtest = x_input[2000:]
    ytest = y_real_label[2000:]

    tokens_tensor_train = tokens_tensor[:2000]
    tokens_tensor_test = tokens_tensor[2000:]
    segments_tensors_train = segments_tensors[:2000]
    segments_tensors_test = segments_tensors[2000:]

    dataloader_train = CustomTextDatasetCombinerBert(tokens_tensor_train, segments_tensors_train, ytrain, xtrain)
    dataloader_train = DataLoader(dataloader_train, batch_size=4, shuffle=True, collate_fn=(lambda x: x))

    dataloader_test = CustomTextDatasetCombinerBert(tokens_tensor_test, segments_tensors_test, ytest, xtest)
    dataloader_test = DataLoader(dataloader_test, batch_size=4, shuffle=False, collate_fn=(lambda x: x))

    model = BertFTLW(device, x_input.shape[1])
    # load_checkpoint('/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/COMBINER-model-D-gating-sel-combiner-epoch149.pt', model)
    model.to(device)
    lr = 1e-4
    optimizer = AdamW(model.parameters(), lr=lr)

    train_loss_set = []
    epochs = 20
    file_path = f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/'
    vect_weights = create_weight_dict(ytrain)
    print(vect_weights)
    loss_fct = CrossEntropyLoss(weight=torch.tensor(vect_weights, dtype=torch.float32), reduction='mean')
    loss_fct.to(device)
    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        model.train()
        pred, true_label = [], []
        for i, batch in enumerate(dataloader_train):
            if i % 30 == 0:
                # break
                print(datetime.fromtimestamp(time.time()), i, train_loss_set[-1] if len(train_loss_set) else 0)
            # continue
            tokens = [torch.Tensor(aa[0]).to(device) for aa in batch]
            masks = [torch.Tensor(aa[1]).to(device) for aa in batch]
            predictions = torch.stack([torch.Tensor(aa[2]) for aa in batch]).to(device)
            outputs = model(tokens, masks, predictions)
            loss = loss_fct(outputs, torch.tensor([aa[3] for aa in batch]).to(device))
            # output = loss(torch.argmax(outputs[1], dim=1).add_(-1), b_labels.add_(-1))
            train_loss_set.append(loss.item())
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()

            pred = pred + outputs.cpu().argmax(1).tolist()
            true_label = true_label + [b[3] for b in batch]
            gc.collect()
            torch.cuda.empty_cache()
        # continue
        print('Results')
        print(classification_report(true_label, pred))
        save_checkpoint(file_path + '/' + f'COMBINER-model-D-gating-sel-combiner-epoch{epoch}-{rel_al}-{rel_b}-{ment}-{s1to10_5}.pt',
                        model, 0)
        with torch.no_grad():
            # validation loop
            model.eval()
            pred, real = [], []
            for i, batch in enumerate(dataloader_test):
                tokens = [torch.Tensor(aa[0]).to(device) for aa in batch]
                masks = [torch.Tensor(aa[1]).to(device) for aa in batch]
                predictions = torch.stack([torch.Tensor(aa[2]) for aa in batch]).to(device)
                outputs = model(tokens, masks, predictions)
                pred = pred + outputs.cpu().argmax(1).tolist()
                real = real + [b[3] for b in batch]
            print(classification_report(real, pred))


def test_with_full_timeline_combiner(rel_al, rel_b, ment, s1to10_5=False):
    import numpy as np
    try:
        # 1/0
        X_text, y_real_label, pred_rel, pred_rel_baixo, pred_1to10, pred_mentions = pickle.load(open('combiner_test_dl.pkl', 'rb'))
    except:
        from sklearn.model_selection import train_test_split
        # 1/0
        X_train, y_train = get_tokens_mask_dates_split(None, None, None, f'Bert_relevancia_baixo_train', 'D', 'relevancia')
        X_test, y_test = get_tokens_mask_dates_split(None, None, None, f'Bert_relevancia_baixo_test', 'D', 'relevancia')
        X_train, X_val_rel_baixo, y_train, y_val_rel_baixo = train_test_split(X_train, y_train, test_size=0.3, random_state=42)


        X_train_1to10, y_train_1to10 = get_tokens_mask_dates_split(None, None, None, f'Bert_ALL_train', 'D', '1to10')
        X_train, X_val_1to10, y_train, y_val_1to10 = train_test_split(X_train_1to10, y_train_1to10, test_size=0.3,
                                                                              random_state=42)

        X_train_rel_am, y_train_rel_am = get_tokens_mask_dates_split(None, None, None, f'Bert_ALL_train', 'D', 'relevancia')
        X_train, X_val_rel_am, y_train, y_val_rel_am = train_test_split(X_train_rel_am, y_train_rel_am, test_size=0.3,
                                                                      random_state=42)

        pred_mentions = prediction_val_mentions()

        pred_rel_baixo = predict_tl(path, X_val_rel_baixo, y_val_rel_baixo, 'all')


        pred_1to10 = predict_tl(path, X_val_1to10, y_val_1to10, 'all')

        pred_rel = predict_tl(path, X_val_rel_am, y_val_rel_am, 'all')

        df_train, _ = load_convert('D', 'all')
        _, X_text, _, y_real_label = train_test_split(df_train['Text'].tolist(), df_train['label'].tolist(), test_size=0.3,
                                                                      random_state=42)
        pickle.dump([X_text, y_real_label, pred_rel, pred_rel_baixo, pred_1to10, pred_mentions], open('combiner_val_dl.pkl', 'wb'))

    models = []
    if rel_al:
        models.append(np.array(pred_rel))
    if rel_b:
        models.append(np.array(pred_rel_baixo))
    if ment:
        models.append(np.array(pred_1to10))
    if s1to10_5:
        models.append(pred_mentions)
    models.append(np.ones(len(pred_rel_baixo)))

    x_input = np.stack(models).T
    texts = ['#'.join(reversed(x)) for x in X_text]
    tokenizer = BertTokenizerFast.from_pretrained("pablocosta/bertweet-br-base-uncased")
    tokenized_texts = tokenizer.batch_encode_plus(texts, max_length=512, padding='max_length',
                                                              truncation=True)
    tokens_tensor = [torch.tensor(t) for t in tokenized_texts['input_ids']]
    segments_tensors = [torch.tensor(t) for t in tokenized_texts['attention_mask']]

    dataloader_test = CustomTextDatasetCombinerBert(tokens_tensor, segments_tensors, y_real_label, x_input)
    dataloader_test = DataLoader(dataloader_test, batch_size=16, shuffle=False, collate_fn=(lambda x: x))

    model = BertFTLW(device, x_input.shape[1])
    model.to(device)
    load_checkpoint(f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/COMBINER-model-D-gating-sel-combiner-epoch-best-{rel_al}-{rel_b}-{ment}-{s1to10_5}.pt', model)
    with torch.no_grad():
        # validation loop
        model.eval()
        pred, real = [], []
        for i, batch in enumerate(dataloader_test):
            tokens = [torch.Tensor(aa[0]).to(device) for aa in batch]
            masks = [torch.Tensor(aa[1]).to(device) for aa in batch]
            predictions = torch.stack([torch.Tensor(aa[2]) for aa in batch]).to(device)
            outputs = model(tokens, masks, predictions)
            pred = pred + outputs.cpu().argmax(1).tolist()
            real = real + [b[3] for b in batch]
        print(classification_report(real, pred))


    return pred, real


def val_with_full_timeline_combiner(rel_al, rel_b, ment, s1to10_5):
    try:
        # 1/0
        X_text, y_real_label, pred_rel, pred_rel_baixo, pred_1to10, pred_mentions = pickle.load(open('combiner_val_dl.pkl', 'rb'))
    except:
        from sklearn.model_selection import train_test_split
        # 1/0
        X_train, y_train = get_tokens_mask_dates_split(None, None, None, f'Bert_relevancia_baixo_train', 'D', 'relevancia')
        X_test, y_test = get_tokens_mask_dates_split(None, None, None, f'Bert_relevancia_baixo_test', 'D', 'relevancia')
        X_train, X_val_rel_baixo, y_train, y_val_rel_baixo = train_test_split(X_train, y_train, test_size=0.3, random_state=42)


        X_train_1to10, y_train_1to10 = get_tokens_mask_dates_split(None, None, None, f'Bert_ALL_train', 'D', '1to10')
        X_train, X_val_1to10, y_train, y_val_1to10 = train_test_split(X_train_1to10, y_train_1to10, test_size=0.3,
                                                                              random_state=42)

        X_train_rel_am, y_train_rel_am = get_tokens_mask_dates_split(None, None, None, f'Bert_ALL_train', 'D', 'relevancia')
        X_train, X_val_rel_am, y_train, y_val_rel_am = train_test_split(X_train_rel_am, y_train_rel_am, test_size=0.3,
                                                                      random_state=42)

        pred_mentions = prediction_val_mentions()


        pred_rel_baixo = predict_tl(path, X_val_rel_baixo, y_val_rel_baixo, 'all')


        pred_1to10 = predict_tl(path, X_val_1to10, y_val_1to10, 'all')

        pred_rel = predict_tl(path, X_val_rel_am, y_val_rel_am, 'all')

        df_train, _ = load_convert('D', 'all')
        _, X_text, _, y_real_label = train_test_split(df_train['Text'].tolist(), df_train['label'].tolist(), test_size=0.3,
                                                                      random_state=42)
        pickle.dump([X_text, y_real_label, pred_rel, pred_rel_baixo, pred_1to10, pred_mentions], open('combiner_val_dl.pkl', 'wb'))

    models = []
    if rel_al:
        models.append(np.array(pred_rel))
    if rel_b:
        models.append(np.array(pred_rel_baixo))
    if ment:
        models.append(np.array(pred_1to10))
    if s1to10_5:
        models.append(pred_mentions)
    models.append(np.ones(len(pred_rel_baixo)))

    x_input = np.stack(models).T
    texts = ['#'.join(reversed(x)) for x in X_text]
    tokenizer = BertTokenizerFast.from_pretrained("pablocosta/bertweet-br-base-uncased")
    tokenized_texts = tokenizer.batch_encode_plus(texts, max_length=512, padding='max_length',
                                                              truncation=True)
    tokens_tensor = [torch.tensor(t) for t in tokenized_texts['input_ids']]
    segments_tensors = [torch.tensor(t) for t in tokenized_texts['attention_mask']]

    dataloader_test = CustomTextDatasetCombinerBert(tokens_tensor, segments_tensors, y_real_label, x_input)
    dataloader_test = DataLoader(dataloader_test, batch_size=16, shuffle=True, collate_fn=(lambda x: x))

    model = BertFTLW(device, x_input.shape[1])
    model.to(device)
    for i in range(1, 20):
        load_checkpoint(f'/home/wesley/Documents/doutorado/doutorado_combinacao_modelos/models/COMBINER-model-D-gating-sel-combiner-epoch{i}-{rel_al}-{rel_b}-{ment}-{s1to10_5}.pt', model)
        with torch.no_grad():
            # validation loop
            model.eval()
            pred, real = [], []
            for i, batch in enumerate(dataloader_test):
                tokens = [torch.Tensor(aa[0]).to(device) for aa in batch]
                masks = [torch.Tensor(aa[1]).to(device) for aa in batch]
                predictions = torch.stack([torch.Tensor(aa[2]) for aa in batch]).to(device)
                outputs = model(tokens, masks, predictions)
                pred = pred + outputs.cpu().argmax(1).tolist()
                real = real + [b[3] for b in batch]
            print(classification_report(real, pred))

def get_tuple(n):
    n = bin(n)[3:]
    # print(n)
    f = [bool(int(v)) for v in n]
    return f


if __name__ == '__main__':
    # rel_am, rel_b, ment, s1to10_5
    # combs = [get_tuple(i) for i in [19, 21, 22, 23, 25, 26,27, 28, 29, 30]]
    #
    # pred_am_b, real_am_b = test_with_full_timeline_combiner(True, True, False, False)
    # pred_b_m, real_b_m = test_with_full_timeline_combiner(False, True, True, False)
    # pred_am_m, real_am_m = test_with_full_timeline_combiner(True, False, True, False)
    # pred_am_b_m, real_am_b_m = test_with_full_timeline_combiner(True, True, True, False)

    combs = [
        # (False, True, True, False),
        # (True, False, True, False),
        # (True, True, False, False),
        (True, True, True, False),
    ]
    for c in combs:
        # train_with_full_timeline_combiner(*c)
        test_with_full_timeline_combiner(*c)


