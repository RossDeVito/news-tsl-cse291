import os
import sys
import random
import torch.optim as optim
from numpy import inf
import os
import random
import sys

import torch.optim as optim
from numpy import inf

sys.path.append(os.path.join("../"))
from torch.utils.data import DataLoader
from date_models.model_utils import *
from date_models.models import *
from datetime import datetime

# Set random seed
seed_val = random.randint(1, 10000)
print('seed:', seed_val)
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def main():
    # Debug toggle
    debug = False
    model_type = 'cnn'  # fcn, deep_fcn, wide_fcn, cnn
    print(model_type)
    # Hyperparams
    hyperparams = \
        {'batch_size': 32,
         'epochs': 1000,
         'learning_rate': 1e-3,
         'early_stop': 10}

    early_stop_count = hyperparams['early_stop']

    # Datasets
    dataset_names = ['crisis', 't17', 'entities']
    print('Training models for the following datasets:', dataset_names)
    ts = datetime.now().strftime('%m%d_%H%M%S')

    print("Run #: ", ts)
    model_dir = '../resources/datewise/{}_date_ranker_{}.all.bin'.format(ts, model_type)
    all_model_dict = {}

    # For each dataset, train and save deep_nn model to dictionary
    for dataset_name in dataset_names:
        # Load the specified dataset
        dataset_path = Path('../datasets/{}'.format(dataset_name))
        dataset = load_dataset(dataset_path)

        # Extract a set of features and labels for all collections (aka topics) in the datase
        if debug:
            print(13 * '+', ' WARNING: DEBUG RUN ', 13 * '+')
            hyperparams['epochs'] = 20
            super_extracted_features, super_dates_y = extract_features_and_labels(dataset.collections[0:1],
                                                                                  dataset_name, method='neural_net')
        else:
            print(13 * '+', ' FULL RUN ', 13 * '+')
            super_extracted_features, super_dates_y = extract_features_and_labels(dataset.collections, dataset_name,
                                                                                  method='neural_net')

        # Run logistic regression for all timelines, in all collections
        super_date_ranker = datewise.SupervisedDateRanker(method='neural_net')
        t, v, f = super_date_ranker.init_data((super_extracted_features, super_dates_y),
                                              return_data=True)  # train/val/test splits

        '''
        Deep net 
        '''
        # Init model
        in_dims = t[0][1].size
        if model_type == 'fcn':
            net = FCNet(in_dims)
        elif model_type == 'deep_fcn':
            net = DeepFCNet(in_dims)
        elif model_type == 'wide_fcn':
            net = WideFCNet(in_dims)
        elif model_type == 'cnn':
            net = CNN(in_dims)
        device = set_device()
        net.to(device)
        net.apply(initialize_weights)

        # Track the best model and loss
        best_loss = inf
        best_net = net

        # Datasets
        dataset_train = DateDataset(t[0], t[1])
        dataset_val = DateDataset(v[0], v[1])
        dataset_test = DateDataset(f[0], f[1])

        dataloader_train = DataLoader(dataset_train, batch_size=hyperparams['batch_size'],
                                      shuffle=True, num_workers=0)
        dataloader_val = DataLoader(dataset_val, batch_size=hyperparams['batch_size'],
                                    shuffle=False, num_workers=0)
        dataloader_test = DataLoader(dataset_test, batch_size=hyperparams['batch_size'],
                                     shuffle=False, num_workers=0)
        # Optimizer
        # optimizer = optim.AdamW(net.parameters(), lr=hyperparams['learning_rate'])
        optimizer = optim.SGD(net.parameters(), lr=hyperparams['learning_rate'])
        criterion = nn.BCELoss()

        # Training
        for epoch in range(hyperparams['epochs']):
            net.train()
            all_loss = []
            for data in dataloader_train:
                optimizer.zero_grad()
                x, y = data
                output = net(data[x].float().to(device))
                output = output.squeeze()
                loss = criterion(output, data[y].float())
                loss.to(device)
                loss.backward()

                optimizer.step()
                all_loss.append(loss.item())
            train_loss = np.mean(all_loss)
            _, _, val_loss = run_validation(net, dataloader_val, criterion, device)
            print("Epoch {0:03d}: Train loss: {1:.3f} | Val loss: {2:.3f}".format(epoch, train_loss, val_loss))
            if val_loss < best_loss:
                best_loss, best_net = val_loss, net
                early_stop_count = hyperparams['early_stop']  # reset early stop counter
            else:
                early_stop_count -= 1
                if early_stop_count <= 0:
                    print(13 * '=', 'STOPPING EARLY!', 13 * '=')
                    break

        # Run val on best model from training
        val_logits, val_labels, _ = run_validation(best_net, dataloader_val, criterion, device)
        run_evaluation(val_logits, val_labels, 'Evaluating on Val Set:')

        # Run on test batch
        test_logits, test_labels, _ = run_validation(best_net, dataloader_test, criterion, device)
        run_evaluation(test_logits, test_labels, 'Evaluating on Test Set:')

        # Store model in dict:
        all_model_dict[dataset_name] = best_net.state_dict()
        print('Stored model for ', dataset_name)

        del net  # clear some memory
        del best_net

    # Save model
    save_model(model_dir, all_model_dict, deep_nn=True)


if __name__ == '__main__':
    print('Starting file: ', os.path.basename(__file__))
    main()
    print('\nCompelted file: ', os.path.basename(__file__))
