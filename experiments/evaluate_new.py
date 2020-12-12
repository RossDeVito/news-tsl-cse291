import argparse

import os
import sys

sys.path.append(os.path.join("../"))

from tilse.evaluation import rouge as tilse_rouge
from pprint import pprint

from date_models.model_utils import *
from pathlib import Path
from tilse.data.timelines import Timeline as TilseTimeline
from tilse.data.timelines import GroundTruth as TilseGroundTruth
from tilse.evaluation import rouge
from news_tls import utils, data, datewise, clust, summarizers
from pprint import pprint

import numpy as np
import time
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

from experiments.metrics.moverscore import get_wordmover_score


def get_scores(metric_desc, pred_tl, groundtruth, evaluator):
    if metric_desc == "concat":
        return evaluator.evaluate_concat(pred_tl, groundtruth)
    elif metric_desc == "agreement":
        return evaluator.evaluate_agreement(pred_tl, groundtruth)
    elif metric_desc == "align_date_costs":
        return evaluator.evaluate_align_date_costs(pred_tl, groundtruth)
    elif metric_desc == "align_date_content_costs":
        return evaluator.evaluate_align_date_content_costs(
            pred_tl, groundtruth)
    elif metric_desc == "align_date_content_costs_many_to_one":
        return evaluator.evaluate_align_date_content_costs_many_to_one(
            pred_tl, groundtruth)


def zero_scores():
    return {'f_score': 0., 'precision': 0., 'recall': 0.}


def evaluate_dates(pred, ground_truth):
    pred_dates = pred.get_dates()
    ref_dates = ground_truth.get_dates()
    shared = pred_dates.intersection(ref_dates)
    n_shared = len(shared)
    n_pred = len(pred_dates)
    n_ref = len(ref_dates)
    prec = n_shared / n_pred
    rec = n_shared / n_ref
    if prec + rec == 0:
        f_score = 0
    else:
        f_score = 2 * prec * rec / (prec + rec)
    return {
        'precision': prec,
        'recall': rec,
        'f_score': f_score,
    }


def date_dist_scores(ref_timeline, ground_truth, p_val=.05):
    '''
    Scores predicted distribution of timeline event dates using 2 sample
        Kolmogorov-Smirnov statistic and the first Wasserstein distance
        (earth mover's distance)

    Returns dict with KS statistic, if time distributions are statistically
        significantly different, and the Wasserstein distance
    '''
    gt_dates = [time.mktime(d.timetuple()) for d in ground_truth.get_dates()]
    ref_dates = [time.mktime(d.timetuple()) for d in ref_timeline.get_dates()]

    scaler = MinMaxScaler()

    gt_scaled = scaler.fit_transform(np.array(gt_dates).reshape(-1, 1)).T[0]
    ref_scaled = scaler.transform(np.array(ref_dates).reshape(-1, 1)).T[0]

    ks_test = stats.ks_2samp(gt_scaled, ref_scaled)
    emd = stats.wasserstein_distance(gt_scaled, ref_scaled)

    # ks_signif 1 when the differentce in date distribution between ground
    #	truth and generated timelines is statistically significant
    return {
        'ks_stat': ks_test.statistic,
        'ks_signif': int(ks_test.pvalue < p_val),
        'earth_movers_distance': emd
    }


def get_average_results(tmp_results):
    rouge_1 = zero_scores()
    rouge_2 = zero_scores()
    date_prf = zero_scores()

    wm_dicts = []
    dd_dicts = []

    for rouge_res, date_res, wm, dd, _ in tmp_results:
        metrics = [m for m in date_res.keys() if m != 'f_score']
        for m in metrics:
            rouge_1[m] += rouge_res['rouge_1'][m]
            rouge_2[m] += rouge_res['rouge_2'][m]
            date_prf[m] += date_res[m]

        wm_dicts.append(wm)
        dd_dicts.append(dd)

    n = len(tmp_results)

    for result in [rouge_1, rouge_2, date_prf]:
        for k in ['precision', 'recall']:
            result[k] /= n
        prec = result['precision']
        rec = result['recall']
        if prec + rec == 0:
            result['f_score'] = 0.
        else:
            result['f_score'] = (2 * prec * rec) / (prec + rec)

    # average wordmover scores
    wm_res = {
        "wordmover": np.mean([d['wordmover'] for d in wm_dicts]),
        "aligned wordmover avg": np.mean(
            [d['aligned wordmover avg'] for d in wm_dicts]
        ),
        "aligned wordmover median": np.mean(
            [d['aligned wordmover median'] for d in wm_dicts]
        ),
        "aligned wordmover std": np.mean(
            [d['aligned wordmover std'] for d in wm_dicts]
        ),
    }

    # average date dist scores
    dd_res = {
        "ks_stat": np.mean([d['ks_stat'] for d in dd_dicts]),
        "ks_signif": np.mean([d['ks_signif'] for d in dd_dicts]),
        "earth_movers_distance": np.mean(
            [d['earth_movers_distance'] for d in dd_dicts]
        ),
    }

    return {
        'rouge 1': rouge_1,
        'rouge 2': rouge_2,
        'date': date_prf,
        'date_dist': dd_res,
        'wordmover': wm_res
    }


def evaluate(tls_model, dataset, result_path, trunc_timelines=False,
             time_span_extension=0, word_mover_stop_words='nltk'):
    results = []
    metric = 'align_date_content_costs_many_to_one'
    evaluator = tilse_rouge.TimelineRougeEvaluator(measures=["rouge_1", "rouge_2"])
    n_topics = len(dataset.collections)

    for i, collection in enumerate(dataset.collections):

        ref_timelines = [TilseTimeline(tl.date_to_summaries)
                         for tl in collection.timelines]
        topic = collection.name
        n_ref = len(ref_timelines)

        if trunc_timelines:
            ref_timelines = data.truncate_timelines(ref_timelines, collection)

        for j, ref_timeline in enumerate(ref_timelines):
            print(f'topic {i + 1}/{n_topics}: {topic}, ref timeline {j + 1}/{n_ref}')

            tls_model.load(ignored_topics=[collection.name])

            ref_dates = sorted(ref_timeline.dates_to_summaries)

            start, end = data.get_input_time_span(ref_dates, time_span_extension)

            collection.start = start
            collection.end = end

            # utils.plot_date_stats(collection, ref_dates)

            l = len(ref_dates)
            k = data.get_average_summary_length(ref_timeline)

            pred_timeline_ = tls_model.predict(
                collection,
                max_dates=l,
                max_summary_sents=k,
                ref_tl=ref_timeline  # only oracles need this
            )

            print('*** PREDICTED ***')
            utils.print_tl(pred_timeline_)

            print('timeline done')
            pred_timeline = TilseTimeline(pred_timeline_.date_to_summaries)
            sys_len = len(pred_timeline.get_dates())
            ground_truth = TilseGroundTruth([ref_timeline])

            rouge_scores = get_scores(
                metric,
                pred_timeline,
                ground_truth,
                evaluator)
            date_scores = evaluate_dates(pred_timeline, ground_truth)
            wm_scores = get_wordmover_score(
                pred_timeline,
                ground_truth,
                word_mover_stop_words,
                device='cpu')
            dd_scores = date_dist_scores(pred_timeline, ground_truth)

            print('sys-len:', sys_len, 'gold-len:', l, 'gold-k:', k)

            print('Alignment-based ROUGE:')
            pprint(rouge_scores)
            print('Date selection:')
            pprint(date_scores)
            pprint(dd_scores)
            print('WordMover scores:')
            pprint(wm_scores)
            print('-' * 100)
            results.append(
                (rouge_scores, date_scores, wm_scores, dd_scores,
                 pred_timeline_.to_dict())
            )

            print("Running average:")
            print(get_average_results(results))
            print()

    avg_results = get_average_results(results)
    print('Average results:')
    pprint(avg_results)
    output = {
        'average': avg_results,
        'results': results,
    }
    utils.write_json(output, result_path)


def main(args):
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f'Dataset not found: {args.dataset}')
    dataset = data.Dataset(dataset_path)
    dataset_name = dataset_path.name

    if args.method == 'datewise':
        resources = Path(args.resources)

        # load date_models for date ranking
        if args.model == 'new_lr':
            method = 'log_regression'
            model_path = resources / 'date_ranker_new_lr.all.pkl'
            key_to_model = utils.load_pkl(model_path)
            model = key_to_model[dataset_name]
        elif is_neural_net(args.model):  # fcn, deep_fcn, cnn, wide_fcn
            model, model_path = model_selector(args.model)
            model.load_state_dict(torch.load(model_path)[dataset_name])
            method = 'neural_net'
            key_to_model = None
        else:
            method = 'linear_regression'
            model_path = resources / 'date_ranker_orig.{}.pkl'.format(dataset_name)
            key_to_model = utils.load_pkl(model_path)
            model = None
        date_ranker = datewise.SupervisedDateRanker(model, method=method)
        sent_collector = datewise.PM_Mean_SentenceCollector(
            clip_sents=5, pub_end=2)
        summarizer = summarizers.CentroidOpt()
        system = datewise.DatewiseTimelineGenerator(
            date_ranker=date_ranker,
            summarizer=summarizer,
            sent_collector=sent_collector,
            key_to_model=key_to_model,
            method=method
        )

    elif args.method == 'clust':
        cluster_ranker = clust.ClusterDateMentionCountRanker()
        clusterer = clust.TemporalMarkovClusterer(max_days=1)
        summarizer = summarizers.CentroidOpt()
        system = clust.ClusteringTimelineGenerator(
            cluster_ranker=cluster_ranker,
            clusterer=clusterer,
            summarizer=summarizer,
            clip_sents=5,
            unique_dates=True,
        )
    elif args.method == 'clust_sbertsum':
        cluster_ranker = clust.ClusterDateMentionCountRanker()
        clusterer = clust.TemporalMarkovClusterer()
        summarizer = summarizers.SBERTSummarizer(
            date_only=False,
            summary_criteria='similarity',
            candidate_sents_per=5,
            compare_with='both'
        )
        system = clust.ClusteringTimelineGenerator(
            cluster_ranker=cluster_ranker,
            clusterer=clusterer,
            summarizer=summarizer,
            clip_sents=5,
            unique_dates=True,
            sbert_summarizer=True
        )
    elif args.method == 'clust_sbert':
        cluster_ranker = clust.ClusterDateMentionCountRanker()
        clusterer = clust.TemporalMarkovClusterer(max_days=365)
        summarizer = summarizers.CentroidOpt()
        system = clust.ClusteringTimelineGenerator(
            clustering_rep='distilroberta-base-paraphrase-v1',
            sbert_sequence_len=512,
            cluster_ranker=cluster_ranker,
            clusterer=clusterer,
            summarizer=summarizer,
            clip_sents=5,
            unique_dates=True,
        )
    elif args.method == 'clust_sbert_sbertsum':
        cluster_ranker = clust.ClusterDateMentionCountRanker()
        clusterer = clust.TemporalMarkovClusterer(max_days=7)
        summarizer = summarizers.SBERTSummarizer(
            date_only=True,
            summary_criteria='similarity',
            candidate_sents_per=5,
            compare_with='both'
        )
        system = clust.ClusteringTimelineGenerator(
            clustering_rep='distilroberta-base-paraphrase-v1',
            sbert_sequence_len=512,
            cluster_ranker=cluster_ranker,
            clusterer=clusterer,
            summarizer=summarizer,
            sbert_summarizer=True,
            clip_sents=5,
            unique_dates=True,
        )
    elif args.method == 'clust_sbert_sbert_old':
        cluster_ranker = clust.ClusterDateMentionCountRanker()
        clusterer = clust.TemporalMarkovClusterer(max_days=7)
        summarizer = summarizers.SubmodularSummarizer()
        system = clust.ClusteringTimelineGenerator(
            clustering_rep='distilroberta-base-paraphrase-v1',
            sbert_sequence_len=512,
            cluster_ranker=cluster_ranker,
            clusterer=clusterer,
            summarizer=summarizer,
            summarizer_rep='same',
            clip_sents=5,
            unique_dates=True,
        )
    elif args.method == 'sbert':
        system = clust.SBERTTimelineGenerator(
            model_name='distilroberta-base-paraphrase-v1',
            cd_n_articles=list(range(5, 20, 1)) + list(range(20, 100, 5)),
            cd_thresholds=np.linspace(.25, .95, 25).tolist(),
            cd_init_max_size=500,
            min_comm_mult=1.5,
            cluster_ranking='date_mention',
            candidate_sents_per=5,
            candidate_articles_per=10,
            similarity_num_articles=10,
            summary_criteria='similarity',
            compare_with='both',
            unique_dates=True
        )
    # elif args.method == 'network':
    #     summarizer = summarizers.CentroidOpt()
    #     system = tr_network.NetworkTimelineGenerator()
    else:
        raise ValueError(f'Method not found: {args.method}')

    if dataset_name == 'entities':
        evaluate(system, dataset, args.output, trunc_timelines=True,
                 time_span_extension=7, word_mover_stop_words='nltk')
    else:
        evaluate(system, dataset, args.output, trunc_timelines=False,
                 time_span_extension=0, word_mover_stop_words='nltk')


if __name__ == '__main__':
    # run examples
    # nohup time python -u evaluate_new.py --dataset "../../data/entities" --method clust --output "../results/cluster_entities_wm.json"
    # nohup time python -u evaluate_new.py --dataset "../../data/entities" --method clust_sbert --output "../results/cluster_sbtfidf_entities_wm.json"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--model', required=False, default='orig')
    parser.add_argument('--resources', default=None,
                        help='model resources for tested method')
    parser.add_argument('--output', default=None)
    main(parser.parse_args())