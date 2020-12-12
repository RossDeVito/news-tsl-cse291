import argparse
import os
import sys

from torch.utils.data import DataLoader

sys.path.append(os.path.join("../"))

from date_models.model_utils import *
from news_tls.data import *


def main(args):
    # Debug toggle
    debug_warning(args.debug)

    # Init
    dataset_names = ['topics']
    # dataset_names = ['crisis', 't17', 'entities']

    print('Evaluating {} models for the following datasets: {}'.format(args.model, dataset_names))

    for dataset_name in dataset_names:

        # Load dataset
        dataset_path = Path('../datasets/{}'.format(dataset_name))
        dataset = load_dataset(dataset_path)

        # Eval orig linear reg model
        orig_eval = False
        if args.model == 'orig':
            orig_eval = True
            method = 'linear_regression'
        elif args.model == 'new_lr':
            method = 'log_regression'
        else:
            method = 'neural_net'

        # Extract a set of features and labels for all collections (aka topics) in the dataset
        super_extracted_features, super_dates_y = extract_features_and_labels(dataset.collections, dataset_name,
                                                                              args.debug, orig_eval, method=method)

        # Run logistic regression for all timelines, in all collections
        super_date_ranker = datewise.SupervisedDateRanker()
        t, v, f = super_date_ranker.init_data((super_extracted_features, super_dates_y),
                                              return_data=True)  # train/val/test splitså

        # Load and run model
        print('Evaluating {} model on a test set from {}:'.format(args.model, dataset_name))
        if is_neural_net(args.model):
            model, model_path = model_selector(args.model)
            model.load_state_dict(torch.load(model_path)[dataset_name])
            dataset_test = DateDataset(f[0], f[1])
            dataloader_test = DataLoader(dataset_test, batch_size=16, shuffle=False, num_workers=0)
            test_logits, test_labels, _ = run_validation(model, dataloader_test)
            run_evaluation(test_logits, test_labels, 'Test Set')
        elif args.model == 'new_lr':
            model_path = '../resources/datewise/date_ranker_new_lr.all.pkl'
            is_valid_path(model_path)
            super_date_ranker.load_model_lr(model_path, dataset_name)
            super_date_ranker.predict_lr(f)


if __name__ == '__main__':
    print('Starting file: ', os.path.basename(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', required=False, default=False)
    parser.add_argument('--model', required=True, default='orig',
                        help='Choose one of the following: orig new_lr deep_fcn wide_fcn fcn cnn')
    main(parser.parse_args())
    print('\nCompelted file: ', os.path.basename(__file__))
