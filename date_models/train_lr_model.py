from date_models.model_utils import *
from news_tls.data import *


def main():
    dataset_names = ['crisis', 't17', 'entities']
    print('Training models for the following datasets:', dataset_names)
    lr_pkl_filename = '../resources/datewise/date_ranker_new_lr.all.pkl'
    all_model_dict = {}

    # For each dataset, train and save lr model to dictionary
    for dataset_name in dataset_names:
        # Load the specified dataset
        dataset_path = Path('../datasets/{}'.format(dataset_name))
        dataset = load_dataset(dataset_path)

        # Extract a set of features and labels for all collections (aka topics) in the dataset
        super_extracted_features, super_dates_y = extract_features_and_labels(dataset.collections, dataset_name,
                                                                              method='neural_net')

        # Run logistic regression for all timelines, in all collections
        super_date_ranker = datewise.SupervisedDateRanker()
        super_date_ranker.init_data((super_extracted_features, super_dates_y))  # train/val/test splits
        super_date_ranker.train_lr()  # train model

        # Save model for dataset
        all_model_dict[dataset_name] = super_date_ranker.get_model()
        print('Stored model for ', dataset_name)

    # Save a dictionary of logistic regression models
    super_date_ranker.save_model(lr_pkl_filename, all_model_dict)


if __name__ == '__main__':
    print('Starting file: ', os.path.basename(__file__))
    main()
    print('\nCompelted file: ', os.path.basename(__file__))
