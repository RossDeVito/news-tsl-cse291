import numpy as np
import datetime
import itertools
import random
import collections
import markov_clustering as mc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse, stats
from typing import List
from news_tls import utils, data

from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import torch


class SBERTTimelineGenerator():
    def __init__(self, 
                 model_name, 
                 sbert_seq_max_len=512,
                 batch_size=128,
                 device='cpu',
                 cd_n_articles=[50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 3],
                 cd_thresholds=[.95, .9, .85, .8, .75, .7, .65, .6, 
                                .55, .5, .45, .4, .35, .3, .2],
                 cd_init_max_size=1000,
                 min_comm_mult=1.05,
                 cluster_ranking='size',
                 candidate_sents_per=5,
                 candidate_articles_per=10,
                 similarity_num_articles=10,
                 summary_criteria='similarity',
                 compare_with='both',
                 unique_dates=True
                 ):
        ''' 
        Create SBERT fast clustering generator 
        
        Args:
            model_name: sentence_transformers model to use
            sbert_seq_max_len: int max length of seqences for SBERT in number
                of tokens
            batch_size: for SBERT encoding
            device: for SBERT encoding. 'cpu', 'cuda', etc

            cd_n_articles: list. community detection n articles to try
            cd_thresholds: list. community detection cos similarity thresholds
                to try
            cd_init_max_size: number of most similar articles unsed in 
                community detection

            min_comm_mult: there must be at least max_dates * min_comm_mult
                communities detected or will reduce criteria
            
            cluster_ranking: ranks resulting clusters from community detection
                'size': ranks clusters by number of articles
                'date_mention': assignes cluster date to be most mentioned date
                    and ranks by mentions of that date

            candidate_sents_per: when generating summary, considers the first
                this number of sentences as candidates and then uses them for
                determining similarity and sentence ranking
            candidate_articles_per: number of most close articles per community
                to use for summarization sentence candidates
            similarity_num_articles: when ranking sentences to use as summary,
                evaluates similaritty with core this number of articles per
                community
            summary_criteria: 'similarity' or 'centroid' for finding best sentences
            compare_with: what sentence summary criteria will be computed with
                'articles': only top similarity_num_articles article vecs
                'sents': only among candiate sentences
                'both': average of both
            unique_dates: if True, only one event per day

        '''
        self.model_name = model_name
        self.sbert = SentenceTransformer(self.model_name)
        self.sbert_seq_max_len = sbert_seq_max_len
        self.sbert.max_seq_length = self.sbert_seq_max_len - 3

        self.batch_size = batch_size
        self.device = device

        self.cd_n_articles = cd_n_articles
        self.cd_thresholds = cd_thresholds
        self.cd_init_max_size = cd_init_max_size

        self.min_comm_mult = min_comm_mult

        self.cluster_ranking = cluster_ranking

        self.candidate_sents_per = candidate_sents_per
        self.candidate_articles_per = candidate_articles_per
        self.similarity_num_articles = similarity_num_articles
        self.summary_criteria = summary_criteria
        self.compare_with = compare_with
        self.unique_dates = unique_dates

    def predict(self, 
                 collection,
                 max_dates=10,
                 max_summary_sents=1,
                 ref_tl=None):
        '''
        Predict timeline for given collection

        Args:
            collection
            max_dates: max number of timeline events
            max_summary_sents: max sentences per timeline event
        '''
        articles = list(collection.articles())
        print("Getting all {} document embeddings...".format(len(articles)))
        full_texts = ['{}. {}'.format(a.title, a.text) for a in articles]
        article_vecs = self.sbert.encode(
            full_texts,
            batch_size=self.batch_size,
            show_progress_bar=True, 
            device=self.device,
            num_workers=24)

        params = list(itertools.product(self.cd_thresholds, self.cd_n_articles))
        print("Detecting communities with {} param options...".format(len(params)))
        n = 0   # for printing n detection runs

        clust_large_enough = []
        clust_too_small = []

        for thresh, min_n in params:
            n += 1
            avg_percentile = np.mean(
                [
                    stats.percentileofscore(self.cd_thresholds, thresh),
                    stats.percentileofscore(self.cd_n_articles, min_n)
                ]
            )
            ordered_clusters = self._community_detection(
                article_vecs,
                thresh, 
                min_n, 
                min(len(article_vecs), self.cd_init_max_size))

            n_clusts = len(ordered_clusters)

            if n_clusts >= (max_dates * self.min_comm_mult):
                print("\tdetecting communities [n={},\tthresh={},\tmin={}\tap={}]".format(
                    n, thresh, min_n, avg_percentile))
                print("\t\t{} communities (enough)".format(n_clusts))
                clust_large_enough.append(
                    (avg_percentile, ordered_clusters)
                )
            else:
                # print("\t\t{} communities (not enough)".format(n_clusts))
                clust_too_small.append(
                    (n_clusts, avg_percentile, ordered_clusters)
                )
        
        if len(clust_large_enough) > 0:
            best = sorted(
                clust_large_enough, 
                key=lambda element: element[0], 
                reverse=True
            )[0]
            ordered_clusters = best[1]
            print("\nUsing {} communities ({} percentile params)\n".format(
                len(ordered_clusters), best[0]
            ))
        else:
            best = sorted(
                clust_too_small, 
                key=lambda element: (element[0], element[1]), 
                reverse=True
            )[0]
            ordered_clusters = best[2]
            print("\nNone with enough communities. Using {} communities ({} percentile params)/n".format(
                best[0], best[1]
            ))

        formated_clusts = []

        if self.cluster_ranking == 'date_mention':
            articles_arr = np.asarray(articles)

            for c in ordered_clusters:
                clust_dict = dict()
                clust_dict['articles'] = articles_arr[c]
                clust_dict['vectors'] = article_vecs[c]

                all_dates = []
                for a in articles_arr[c]:
                    all_dates.append( a.time.date() )
                    for s in a.sentences:
                        if s.get_date():
                            all_dates.append(s.get_date())

                most_common = collections.Counter(all_dates).most_common(1)[0]
                clust_dict['date'] = most_common[0]
                clust_dict['date_count'] = most_common[1]
                formated_clusts.append(clust_dict)

            formated_clusts = sorted(formated_clusts, 
                                        key=lambda c: (c['date_count'], 
                                                        len(c['articles'])),
                                        reverse=True)

        elif self.cluster_ranking == 'size':
            articles_arr = np.asarray(articles)
            for c in ordered_clusters:
                clust_dict = dict()
                clust_dict['articles'] = articles_arr[c]
                clust_dict['vectors'] = article_vecs[c]
                clust_dict['date'] = None
                clust_dict['date_count'] = None
                formated_clusts.append(clust_dict)
        else:
            raise ValueError("invalid cluster_ranking option")

        
        print('summarization...')
        sys_l = 0
        sys_m = 0
        ref_m = max_dates * max_summary_sents

        date_to_summary = collections.defaultdict(list)

        for c in formated_clusts:
            if c['date']:
                print('\n\tcommunity with {} articles and {} date count'.format(
                    len(c['articles']), c['date_count']))
                core_doc_vecs = c['vectors'][:self.similarity_num_articles]
                candidate_sents = []
                date_docs = []

                for a in c['articles']:
                    start_ind = 0
                    article_added = False

                    if a.time.date() == c['date']:
                        date_docs.append(a)
                        article_added = True
                        start_ind = self.candidate_sents_per
                        for s in a.sentences[:start_ind]:
                            candidate_sents.append(s)
                    
                    for s in a.sentences[start_ind:]:
                        if s.get_date() and s.get_date() == c['date']:
                            if not article_added:
                                date_docs.append(a)
                                article_added = True
                            candidate_sents.append(s)

                if len(candidate_sents) == 0:
                    print("no date linked candidate sentences")
                    continue

                print("...encoding candidate sentences...")
                candidate_sents_text = [s.raw for s in candidate_sents]
                candidate_sents_vecs = self.sbert.encode(
                    candidate_sents_text,
                    batch_size=self.batch_size,
                    show_progress_bar=True, 
                    device=self.device,
                    num_workers=24)

                if self.summary_criteria == 'centroid':
                    doc_compare_vecs = np.mean(core_doc_vecs, axis=0)
                    assert len(doc_compare_vecs) == len(core_doc_vecs[0])

                    sent_compare_vecs = np.mean(candidate_sents_vecs, axis=0)
                    assert len(doc_compare_vecs) == len(sent_compare_vecs)
                else:
                    doc_compare_vecs = core_doc_vecs
                    sent_compare_vecs = candidate_sents_vecs

                if self.compare_with == 'both':
                    doc_sim = np.mean(
                        util.pytorch_cos_sim(
                                candidate_sents_vecs, 
                                torch.from_numpy(doc_compare_vecs).float()
                            ).numpy(), 
                        axis=1)
                    sent_sim = np.mean(
                        util.pytorch_cos_sim(
                                candidate_sents_vecs, 
                                torch.from_numpy(sent_compare_vecs).float()
                            ).numpy(), 
                        axis=1)

                    sent_scores =  np.mean(np.stack((doc_sim, sent_sim)), axis=0)
                else:
                    raise NotImplementedError

                top_sent_inds = np.argsort(-sent_scores)[:max_summary_sents]

                event_summary = ''
                date = c['date']
                for ind in top_sent_inds:
                    event_summary += candidate_sents_text[ind] + ' '
                if not date:
                    print('\tNo date for event found')
                    continue
                if self.unique_dates and date in date_to_summary:
                    print('\tSkipping repeat date')
                    continue

                date_to_summary[date] += [event_summary]
                print('\t\t{}\t{}'.format(date, event_summary))

                sys_m += max_summary_sents
                if self.unique_dates:
                    sys_l += 1

                if sys_m >= ref_m or sys_l >= max_dates:
                    break

            else:
                print('\n\tcommunity with {} articles'.format(len(c['articles'])))
                core_doc_vecs = c['vectors'][:self.similarity_num_articles]
                core_articles = c['articles'][:self.candidate_articles_per]
                candidate_sents = [
                    s for a in core_articles for s in a.sentences[:self.candidate_sents_per]
                ]
                candidate_sents_text = [s.raw for s in candidate_sents]
                candidate_sents_vecs = self.sbert.encode(
                    candidate_sents_text,
                    batch_size=self.batch_size,
                    show_progress_bar=True, 
                    device=self.device,
                    num_workers=24)

                if self.summary_criteria == 'centroid':
                    doc_compare_vecs = np.mean(core_doc_vecs, axis=0)
                    assert len(doc_compare_vecs) == len(core_doc_vecs[0])

                    sent_compare_vecs = np.mean(candidate_sents_vecs, axis=0)
                    assert len(doc_compare_vecs) == len(sent_compare_vecs)
                else:
                    doc_compare_vecs = core_doc_vecs
                    sent_compare_vecs = candidate_sents_vecs

                if self.compare_with == 'both':
                    doc_sim = np.mean(
                        util.pytorch_cos_sim(
                                candidate_sents_vecs, 
                                torch.from_numpy(doc_compare_vecs).float()
                            ).numpy(), 
                        axis=1)
                    sent_sim = np.mean(
                        util.pytorch_cos_sim(
                                candidate_sents_vecs, 
                                torch.from_numpy(sent_compare_vecs).float()
                            ).numpy(), 
                        axis=1)

                    sent_scores =  np.mean(np.stack((doc_sim, sent_sim)), axis=0)

                top_sent_inds = np.argsort(-sent_scores)[:max_summary_sents]

                event_summary = ''
                date = None
                for ind in top_sent_inds:
                    event_summary += candidate_sents_text[ind] + ' '
                    if not date:
                        if candidate_sents[ind].get_date():
                            date = candidate_sents[ind].get_date()
                        else:
                            date = candidate_sents[ind].pub_time.date()
                if not date:
                    print('\tNo date for event found')
                    continue
                if self.unique_dates and date in date_to_summary:
                    print('\tSkipping repeat date')
                    continue

                date_to_summary[date] += [event_summary]
                print('\t\t{}\t{}'.format(date, event_summary))

                sys_m += max_summary_sents
                if self.unique_dates:
                    sys_l += 1

                if sys_m >= ref_m or sys_l >= max_dates:
                    break

        timeline = []
        for d, summary in date_to_summary.items():
            t = datetime.datetime(d.year, d.month, d.day)
            timeline.append((t, summary))
        timeline.sort(key=lambda x: x[0])

        return data.Timeline(timeline)

    def _community_detection(self, 
            embeddings, 
            threshold=0.75, 
            min_community_size=10, 
            init_max_size=1000):
        '''
        Function for Fast Community Detection from 
            https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/clustering/fast_clustering.py
        
        Finds in the embeddings all communities, i.e. embeddings that are close (closer than threshold).
        Returns only communities that are larger than min_community_size. The communities are returned
        in decreasing order. The first element in each list is the central point in the community.
        '''

        # Compute cosine similarity scores
        cos_scores = util.pytorch_cos_sim(embeddings, embeddings)

        # Minimum size for a community
        top_k_values, _ = cos_scores.topk(k=min_community_size, largest=True)

        # Filter for rows >= min_threshold
        extracted_communities = []
        for i in range(len(top_k_values)):
            if top_k_values[i][-1] >= threshold:
                new_cluster = []

                # Only check top k most similar entries
                top_val_large, top_idx_large = cos_scores[i].topk(k=init_max_size, largest=True)
                top_idx_large = top_idx_large.tolist()
                top_val_large = top_val_large.tolist()

                if top_val_large[-1] < threshold:
                    for idx, val in zip(top_idx_large, top_val_large):
                        if val < threshold:
                            break

                        new_cluster.append(idx)
                else:
                    # Iterate over all entries (slow)
                    for idx, val in enumerate(cos_scores[i].tolist()):
                        if val >= threshold:
                            new_cluster.append(idx)

                extracted_communities.append(new_cluster)

        # Largest cluster first
        extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)

        # Step 2) Remove overlapping communities
        unique_communities = []
        extracted_ids = set()

        for community in extracted_communities:
            add_cluster = True
            for idx in community:
                if idx in extracted_ids:
                    add_cluster = False
                    break

            if add_cluster:
                unique_communities.append(community)
                for idx in community:
                    extracted_ids.add(idx)

        return unique_communities

    def load(self, ignored_topics):
        pass

class ClusteringTimelineGenerator():
    def __init__(self,
                 clustering_rep='tfidf', #'distilroberta-base-paraphrase-v1'
                 sbert_sequence_len=None,
                 clusterer=None,
                 cluster_ranker=None,
                 summarizer=None,
                 summarizer_rep='tfidf',
                 sbert_summarizer=False,
                 clip_sents=5,
                 key_to_model=None,
                 unique_dates=True):

        '''
        args:
            clustering_rep: one of following options for text representation
                to be used by clusterer
                    'tfidf': uses tfidf vectorizer w/ stopwords from original
                        paper
                    model_name: string to be used to load pretrained SBERT
                        model from Sentence-Transformers if sbert_sequence_len
                        is not None
            sbert_sequence_len: int max length of tokens for sbert to use 
                (e.g. 512, 128)
            summarizer_rep: vector embedding used to represent sentence when
                creating summary
                    'tfidf': use tfidf like original
                    'same': use clustering SBERT model
            sbert_summarizer: True if using SBERTSummarizer
        '''
        self.clustering_rep = clustering_rep
        self.sbert_sequence_len = sbert_sequence_len
        self.clusterer = clusterer or TemporalMarkovClusterer()
        self.cluster_ranker = cluster_ranker or ClusterDateMentionCountRanker()
        self.summarizer = summarizer or summarizers.CentroidOpt()
        self.summarizer_rep = summarizer_rep
        self.sbert_summarizer = sbert_summarizer
        self.key_to_model = key_to_model
        self.unique_dates = unique_dates
        self.clip_sents = clip_sents

    def predict(self,
                collection,
                max_dates=10,
                max_summary_sents=1,
                ref_tl=None,
                input_titles=False,
                output_titles=False,
                output_body_sents=True):

        print('clustering articles...')
        if self.clustering_rep == 'tfidf':
            print("\tusing tfidf")
            doc_vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
            clusters = self.clusterer.cluster(collection, doc_vectorizer)
        # use sentence transformer
        elif self.sbert_sequence_len:
            print("\tusing {} with {} max tokens".format(
                self.clustering_rep, 
                self.sbert_sequence_len))
            sbert_model = SentenceTransformer(self.clustering_rep)
            sbert_model.max_seq_length = self.sbert_sequence_len - 3
            clusters = self.clusterer.cluster(collection, sbert_model, sbert=True)
        else:
            raise NotImplementedError("invalid clustering_rep and sbert_sequence_len combination")   


        print('assigning cluster times...')
        for c in clusters:
            c.time = c.most_mentioned_time()
            if c.time is None:
                c.time = c.earliest_pub_time()

        print('ranking clusters...')
        ranked_clusters = self.cluster_ranker.rank(clusters, collection)

        if self.sbert_summarizer:
            print('using a SBERTSummarizer')
        elif self.summarizer_rep == 'tfidf':
            print('tfidf vectorizing sentences...')
            raw_sents = [s.raw for a in collection.articles() for s in
                        a.sentences[:self.clip_sents]]
            vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
            vectorizer.fit(raw_sents)

            using_sbert = False
        elif self.summarizer_rep == 'same':
            print("\reusing clustering sbert model")
            vectorizer = sbert_model
            using_sbert = True

        def sent_filter(sent):
            """
            Returns True if sentence is allowed to be in a summary.
            """
            lower = sent.raw.lower()
            if not any([kw in lower for kw in collection.keywords]):
                return False
            elif not output_titles and sent.is_title:
                return False
            elif not output_body_sents and not sent.is_sent:
                return False
            else:
                return True

        print('summarization...')
        sys_l = 0
        sys_m = 0
        ref_m = max_dates * max_summary_sents

        date_to_summary = collections.defaultdict(list)
        for c in ranked_clusters:
            date = c.time.date()
            if self.sbert_summarizer:
                    summary = self.summarizer.summarize(
                        c.articles,
                        k=max_summary_sents,
                        date=date
                    )
            else:
                c_sents = self._select_sents_from_cluster(c)
                #print("C", date, len(c_sents), "M", sys_m, "L", sys_l)
                summary = self.summarizer.summarize(
                    c_sents,
                    k=max_summary_sents,
                    vectorizer=vectorizer,
                    filter=sent_filter,
                    sbert=using_sbert
                )

            if summary:
                if self.unique_dates and date in date_to_summary:
                    continue
                date_to_summary[date] += summary
                sys_m += len(summary)
                if self.unique_dates:
                    sys_l += 1

            if sys_m >= ref_m or sys_l >= max_dates:
                break

        timeline = []
        for d, summary in date_to_summary.items():
            t = datetime.datetime(d.year, d.month, d.day)
            timeline.append((t, summary))
        timeline.sort(key=lambda x: x[0])

        return data.Timeline(timeline)

    def _select_sents_from_cluster(self, cluster):
        sents = []
        for a in cluster.articles:
            pub_d = a.time.date()
            for s in a.sentences[:self.clip_sents]:
                sents.append(s)
        return sents

    def load(self, ignored_topics):
        pass


################################# CLUSTERING ###################################


class Cluster:
    def __init__(self, articles, vectors, centroid, time=None, id=None):
        self.articles = sorted(articles, key=lambda x: x.time)
        self.centroid = centroid
        self.id = id
        self.vectors = vectors
        self.time = time

    def __len__(self):
        return len(self.articles)

    def pub_times(self):
        return [a.time for a in self.articles]

    def earliest_pub_time(self):
        return min(self.pub_times())

    def most_mentioned_time(self):
        mentioned_times = []
        for a in self.articles:
            for s in a.sentences:
                if s.time and s.time_level == 'd':
                    mentioned_times.append(s.time)
        if mentioned_times:
            return collections.Counter(mentioned_times).most_common()[0][0]
        else:
            return None

    def update_centroid(self):
        X = sparse.vstack(self.vectors)
        self.centroid = sparse.csr_matrix.mean(X, axis=0)


class Clusterer():
    def cluster(self, collection, vectorizer) -> List[Cluster]:
        raise NotImplementedError


class OnlineClusterer(Clusterer):
    def __init__(self, max_days=1, min_sim=0.5):
        self.max_days = max_days
        self.min_sim = min_sim

    def cluster(self, collection, vectorizer, sbert=False) -> List[Cluster]:
        # build article vectors
        texts = ['{} {}'.format(a.title, a.text) for a in collection.articles]
        try:
            X = vectorizer.transform(texts)
        except:
            X = vectorizer.fit_transform(texts)

        id_to_vector = {}
        for a, x in zip(collection.articles(), X):
            id_to_vector[a.id] = x

        online_clusters = []

        for t, articles in collection.time_batches():
            for a in articles:

                # calculate similarity between article and all clusters
                x = id_to_vector[a.id]
                cluster_sims = []
                for c in online_clusters:
                    if utils.days_between(c.time, t) <= self.max_days:
                        centroid = c.centroid
                        sim = cosine_similarity(centroid, x)[0, 0]
                        cluster_sims.append(sim)
                    else:
                        cluster_sims.append(0)

                # assign article to most similar cluster (if over threshold)
                cluster_found = False
                if len(online_clusters) > 0:
                    i = np.argmax(cluster_sims)
                    if cluster_sims[i] >= self.min_sim:
                        c = online_clusters[i]
                        c.vectors.append(x)
                        c.articles.append(a)
                        c.update_centroid()
                        c.time = t
                        online_clusters[i] = c
                        cluster_found = True

                # initialize new cluster if no cluster was similar enough
                if not cluster_found:
                    new_cluster = Cluster([a], [x], x, t)
                    online_clusters.append(new_cluster)

        clusters = []
        for c in online_clusters:
            cluster = Cluster(c.articles, c.vectors)
            clusters.append(cluster)

        return clusters


class TemporalMarkovClusterer(Clusterer):
    def __init__(self, max_days=1):
        self.max_days = max_days

    def cluster(self, collection, vectorizer, sbert=False) -> List[Cluster]:
        articles = list(collection.articles())
        texts = ['{} {}'.format(a.title, a.text) for a in articles]
        if sbert:
            X = vectorizer.encode(
                    texts,
                    batch_size=128,
                    show_progress_bar=True, 
                    device='cpu',
                    num_workers=24)
        else:
            try:
                X = vectorizer.transform(texts)
            except:
                X = vectorizer.fit_transform(texts)

        times = [a.time for a in articles]

        print('temporal graph...')
        S = self.temporal_graph(X, times)
        #print('S shape:', S.shape)
        print('run markov clustering...')
        result = mc.run_mcl(S)
        print('done')

        idx_clusters = mc.get_clusters(result)
        idx_clusters.sort(key=lambda c: len(c), reverse=True)

        print(f'times: {len(set(times))} articles: {len(articles)} '
              f'clusters: {len(idx_clusters)}')

        clusters = []
        for c in idx_clusters:
            c_vectors = [X[i] for i in c]
            c_articles = [articles[i] for i in c]

            if sparse.issparse(c_vectors):
                Xc = sparse.vstack(c_vectors)
                centroid = sparse.csr_matrix(Xc.mean(axis=0))
            else:
                Xc = np.vstack(c_vectors)
                centroid = np.mean(Xc, axis=0)
            cluster = Cluster(c_articles, c_vectors, centroid=centroid)
            clusters.append(cluster)

        return clusters

    def temporal_graph(self, X, times):
        times = [utils.strip_to_date(t) for t in times]
        time_to_ixs = collections.defaultdict(list)
        for i in range(len(times)):
            time_to_ixs[times[i]].append(i)

        n_items = X.shape[0]
        S = sparse.lil_matrix((n_items, n_items))
        start, end = min(times), max(times)
        total_days = (end - start).days + 1

        for n in range(total_days + 1):
            t = start + datetime.timedelta(days=n)
            window_size =  min(self.max_days + 1, total_days + 1 - n)
            window = [t + datetime.timedelta(days=k) for k in range(window_size)]

            if n == 0 or len(window) == 1:
                indices = [i for t in window for i in time_to_ixs[t]]
                if len(indices) == 0:
                    continue

                if sparse.issparse(X):
                    X_n = sparse.vstack([X[i] for i in indices])
                else:
                    X_n = np.vstack([X[i] for i in indices])

                S_n = cosine_similarity(X_n)
                n_items = len(indices)
                for i_x, i_n in zip(indices, range(n_items)):
                    for j_x, j_n in zip(indices, range(i_n + 1, n_items)):
                        S[i_x, j_x] = S_n[i_n, j_n]
            else:
                # prev is actually prev + new
                prev_indices = [i for t in window for i in time_to_ixs[t]]
                new_indices = time_to_ixs[window[-1]]

                if len(new_indices) == 0:
                    continue

                if sparse.issparse(X):
                    X_prev = sparse.vstack([X[i] for i in prev_indices])
                    X_new = sparse.vstack([X[i] for i in new_indices])
                else:
                    X_prev = np.vstack([X[i] for i in prev_indices])
                    X_new = np.vstack([X[i] for i in new_indices])

                S_n = cosine_similarity(X_prev, X_new)
                n_prev, n_new = len(prev_indices), len(new_indices)
                for i_x, i_n in zip(prev_indices, range(n_prev)):
                    for j_x, j_n in zip(new_indices, range(n_new)):
                        S[i_x, j_x] = S_n[i_n, j_n]

        return sparse.csr_matrix(S)


############################### CLUSTER RANKING ################################


class ClusterRanker:
    def rank(self, clusters, collection, vectorizer):
        raise NotImplementedError


class ClusterSizeRanker(ClusterRanker):
    def rank(self, clusters, collection=None, vectorizer=None):
        return sorted(clusters, key=len, reverse=True)


class ClusterDateMentionCountRanker(ClusterRanker):
    def rank(self, clusters, collection=None, vectorizer=None):
        date_to_count = collections.defaultdict(int)
        for a in collection.articles():
            for s in a.sentences:
                d = s.get_date()
                if d:
                    date_to_count[d] += 1

        clusters = sorted(clusters, reverse=True, key=len)

        def get_count(c):
            t = c.most_mentioned_time()
            if t:
                return date_to_count[t.date()]
            else:
                return 0

        clusters = sorted(clusters, reverse=True, key=get_count)
        return sorted(clusters, key=len, reverse=True)






