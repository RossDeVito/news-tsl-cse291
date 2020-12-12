'''
Utils for date prediction models
'''
import pickle
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import normalize
from tilse.data.timelines import Timeline as TilseTimeline

from date_models.models import *
from date_models.models import FCNet, DeepFCNet, WideFCNet, CNN
from news_tls import datewise
from news_tls.data import Dataset


class DateDataset(Dataset):
    def __init__(self, x, y):
        self.features = normalize(x, norm='l2', axis=0)
        self.labels = list(map(int, y))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample = {'feature': self.features[idx], 'label': self.labels[idx]}
        return sample


def debug_warning(debug):
    if debug:
        print('\n' + '*' * 10 + ' WARNING, DEBUG MODE ' + '*' * 10)
    else:
        print('\n' + '*' * 10 + ' FULL RUN ' + '*' * 10)


def metrics(y_true, y_pred, linear=False):
    '''Calculate accuracy, precision, recall, f1 macro, f1 micro'''
    if linear:
        threshold = 100
        y_pred = np.asarray(y_pred)
        y_pred = np.where(y_pred > threshold, 1, 0).astype(str)
    a = accuracy_score(y_true, y_pred)
    p_mi = precision_score(y_true, y_pred, average='micro')
    p_ma = precision_score(y_true, y_pred, average='macro')
    r_mi = recall_score(y_true, y_pred, average='micro')
    r_ma = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    print("Accuracy: {}, P-mi: {}, R-mi: {}, F1-Macro: {}, P-ma: {}, R-ma: {}, F1-Micro: {}".format(a, p_mi, r_mi,
                                                                                                    f1_micro, p_ma,
                                                                                                    r_ma, f1_macro),
          end='\n\n')


def load_dataset(path):
    dataset = Dataset(path)
    return dataset


def set_device():
    if torch.cuda.is_available():
        print("Nice! Using GPU.")
        return 'cuda'
    else:
        print("Watch predictions! Using CPU.")
        return 'cpu'


def precision_recall_f1(y_true, y_pred, linear=False):
    if linear:
        threshold = 100
        y_pred = np.asarray(y_pred)
        y_pred = np.where(y_pred > threshold, 1, 0).astype(str)
    s = {'p_mi': precision_score(y_true, y_pred, average='micro'),
         'r_mi': recall_score(y_true, y_pred, average='micro'),
         'f1_mi': f1_score(y_true, y_pred, average='micro'),
         'p_ma': precision_score(y_true, y_pred, average='macro'),
         'r_ma': recall_score(y_true, y_pred, average='macro'),
         'f1_ma': f1_score(y_true, y_pred, average='macro')}
    print('|micro| p: {:4f} r: {:4f} f1: {:4f} |macro| p: {:4f} r: {:4f} f1: {:4f}'.format(s['p_mi'], s['r_mi'],
                                                                                           s['f1_mi'], s['p_ma'],
                                                                                           s['r_ma'], s['f1_ma']))


def inspect(df):
    print("HEAD:\n", df.head())
    print("DESCRIBE:\n", df.describe())
    print("COLS:\n", df.columns)
    print("ALL:\n", df)


def save_model(filename, model_dict, neural_net=False):
    '''
    Pickles the passed model into the filename provided
    '''
    if neural_net:
        torch.save(model_dict, filename)
    else:
        log_regression_model_pkl = open(filename, 'wb')
        pickle.dump(model_dict, log_regression_model_pkl)
        log_regression_model_pkl.close()
    print('File saved: ', filename)


def flat_accuracy(preds, labels):
    return np.sum(preds == labels) / len(labels)


def extract_features_and_labels(collections, dataset_name, debug=False, orig_eval=False, method='linear_regression',
                                n_features=7):
    '''
    A function that extracts all the date features and labels for a collection of
    topics in a dataset.
    Inputs:
        collections: a collections object containing all topics in the dataset
        dataset_name: the name of the dataset (t17, crisis, or entities)
        n_features: number of extracted date features to use in the model
    Outputs:
        super_extracted_features: numpy array of all date features (n_dates x n_features)
        super_dates_y: list of labels corresponding to the extracted features.
            0 = not important date, 1 = important date
    '''
    if debug:
        collections = collections[0:1]  # shortened run
    super_extracted_features, super_dates_y = np.empty([0, n_features]), []
    n_topics = len(collections)
    ys, y_preds = [], []
    for i, collection in enumerate(collections):
        print('{}/{}: Extracting features from: ({}:{})'.format(i + 1, n_topics, dataset_name, collection.name))
        important_dates = set()
        dates_y = []

        # Get a list of each timeline in current collection
        ref_timelines = [TilseTimeline(tl.date_to_summaries)
                         for tl in collection.timelines]

        # Init supervised data ranker and extract all dates and their corresponding features
        date_ranker = datewise.SupervisedDateRanker(method=method)
        dates, extracted_features = date_ranker.extract_features(collection)

        # Get all ground-truth important dates
        for timeline in ref_timelines:
            important_dates = important_dates.union(timeline.dates_to_summaries.keys())

        # Convert important dates to binary labels. 1=important, 0=not important
        for date in dates:
            dates_y.append('1') if date in important_dates else dates_y.append('0')

        if orig_eval:
            model_path = '../resources/datewise/date_ranker_orig.{}.pkl'.format(dataset_name)
            date_ranker.init_data((extracted_features, dates_y))
            date_ranker.load_model_orig(model_path, collection.name)
            y, y_pred = date_ranker.predict_lr(linear=True)
            ys.append(y)
            y_preds.append(y_pred)

        # Append extracted features and labels from collection to a super set of features and labels
        super_extracted_features = np.concatenate((super_extracted_features, extracted_features), axis=0)
        super_dates_y.append(dates_y)

    if orig_eval:
        ys_flat_list = [item for sublist in ys for item in sublist]
        y_preds_flat_list = [item for sublist in y_preds for item in sublist]
        precision_recall_f1(ys_flat_list, y_preds_flat_list, linear=True)
        metrics(ys_flat_list, y_preds_flat_list, linear=True)
    super_dates_y = sum(super_dates_y, [])  # flatten list
    return super_extracted_features, super_dates_y


def run_validation(net, dataloader, criterion=None, device='cpu'):
    net.eval()
    if criterion is None:
        criterion = nn.BCELoss()
    all_logits, all_loss, all_labels = [], [], []
    for data in dataloader:
        x, y = data
        net.zero_grad()
        with torch.no_grad():
            output = net(data[x].float())
        output = output.squeeze()
        loss = criterion(output, data[y].float())
        all_loss.append(loss.item())
        all_logits.append(output.detach().cpu().numpy())
        all_labels.append(data[y])
    mean_loss = np.mean(all_loss)
    return all_logits, all_labels, mean_loss


def run_evaluation(test_logits, test_labels, title):
    # Calc accuracy, p, r, f1
    threshold = 0.5
    test_p1 = np.concatenate(np.asarray(test_logits), axis=0)
    # test_p2 = np.argmax(test_p1, axis=1).flatten()
    test_p2 = np.where(test_p1 > threshold, 1, 0).flatten()
    test_l1 = np.concatenate(test_labels, axis=0)
    test_l2 = test_l1.flatten()
    a = flat_accuracy(test_p2, test_l2)
    print(f"Results: {title}")
    print(f"Acc: {a:.4f}")
    precision_recall_f1(test_l2, test_p2)


def ave_results(results):
    a = mean(results['a'])
    p = mean(results['p'])
    r = mean(results['r'])
    f1_macro = mean(results['f1_macro'])
    f1_micro = mean(results['f1_micro'])
    print(
        'Averages from orig run >> a: {}, p: {}, r: {}, f1_macro: {}, f1_micro: {}'.format(a, p, r, f1_macro, f1_micro))


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def is_neural_net(model_name):
    if model_name in ['fcn', 'wide_fcn', 'deep_fcn', 'cnn']:
        return True
    else:
        return False


def is_valid_path(model_path):
    if not Path(model_path).exists():
        raise FileNotFoundError(f'Model not found: {model_path}')


def model_selector(model_name):
    path = '../resources/datewise/date_ranker_{}.all.bin'.format(model_name)
    is_valid_path(path)
    if model_name == 'fcn':
        return FCNet(), path
    if model_name == 'wide_fcn':
        return WideFCNet(), path
    if model_name == 'deep_fcn':
        return DeepFCNet(), path
    if model_name == 'cnn':
        return CNN(), path
    raise Exception('Neural net model not recognized.')
