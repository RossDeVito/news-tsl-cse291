import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from date_models.model_utils import *
from news_tls.data import *


def main(args):
    # Debug toggle
    debug_warning(args.debug)

    # Init
    dataset_names = ['crisis', 't17', 'entities']
    print('Evaluating {} models for the following datasets: {}'.format(args.model, dataset_names))

    for dataset_name in dataset_names:

        # Get saved model's path
        version, model_file_type, orig_eval = 'all', 'pkl', False
        if args.model == 'deep_nn': model_file_type = 'bin'
        if args.model == 'orig':
            version = dataset_name
            orig_eval = True
        model_path = '../resources/datewise/date_ranker_{}.{}.{}'.format(args.model, version, model_file_type)

        if not Path(model_path).exists():
            raise FileNotFoundError(f'Model not found: {model_path}')

        # Load dataset
        dataset_path = Path('../datasets/{}'.format(dataset_name))
        dataset = load_dataset(dataset_path)

        # Extract a set of features and labels for all collections (aka topics) in the dataset
        if args.debug:
            super_extracted_features, super_dates_y = extract_features_and_labels(dataset.collections[0:1],
                                                                                  dataset_name, orig_eval)
        else:
            super_extracted_features, super_dates_y = extract_features_and_labels(dataset.collections,
                                                                                  dataset_name, orig_eval)

        # Run logistic regression for all timelines, in all collections
        super_date_ranker = datewise.SupervisedDateRanker()
        t, v, f = super_date_ranker.init_data((super_extracted_features, super_dates_y),
                                              return_data=True)  # train/val/test splits

        # Load and run model
        print('Evaluating {} model on a test set from {}:'.format(args.model, dataset_name))
        if args.model == 'deep_nn':
            in_dims = t[0][1].size
            model = FCNet(in_dims)
            model.load_state_dict(torch.load(model_path)[dataset_name])
            dataset_test = DateDataset(f[0], f[1])
            dataloader_test = DataLoader(dataset_test, batch_size=16, shuffle=False, num_workers=0)
            test_logits, test_labels, _ = run_validation(model, dataloader_test)
            run_evaluation(test_logits, test_labels, 'Test Set')
        elif args.model == 'new_lr':
            super_date_ranker.load_model_lr(model_path, dataset_name)
            super_date_ranker.predict_lr(f)


if __name__ == '__main__':
    print('Starting file: ', os.path.basename(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', required=False, default=False)
    parser.add_argument('--model', required=True, default='orig',
                        help='model to use for date predictions. "orig" for original, '
                             '"new_lr" for new logistic regression, and'
                             '"deep_nn for deep neural net.')  # orig, new_lr, deep_nn
    main(parser.parse_args())
    print('\nCompelted file: ', os.path.basename(__file__))
