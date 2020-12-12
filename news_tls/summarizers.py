import collections

import networkx as nx
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans

from sentence_transformers import SentenceTransformer, util
import torch


class Summarizer:

    def summarize(self, sents, k, vectorizer, filters=None):
        raise NotImplementedError


class TextRank(Summarizer):

    def __init__(self, max_sim=0.9999):
        self.name = 'TextRank Summarizer'
        self.max_sim = max_sim

    def score_sentences(self, X):
        S = cosine_similarity(X)
        nodes = list(range(S.shape[0]))
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        for i in range(S.shape[0]):
            for j in range(S.shape[0]):
                graph.add_edge(nodes[i], nodes[j], weight=S[i, j])
        pagerank = nx.pagerank(graph, weight='weight')
        scores = [pagerank[i] for i in nodes]
        return scores

    def summarize(self, sents, k, vectorizer, filter=None, sbert=False):
        raw_sents = [s.raw for s in sents]
        if sbert:
            X = vectorizer.encode(
                    raw_sents,
                    batch_size=128,
                    show_progress_bar=True, 
                    device='cpu',
                    num_workers=24)
        else:
            try:
                X = vectorizer.transform(raw_sents)
            except:
                return None
        

        scores = self.score_sentences(X)

        indices = list(range(len(sents)))
        ranked = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)

        summary_sents = []
        summary_vectors = []
        for i, _ in ranked:
            if len(summary_sents) >= k:
                break
            new_x = X[i]
            s = sents[i]
            is_redundant = False
            for x in summary_vectors:
                if cosine_similarity(new_x, x)[0, 0] > self.max_sim:
                    is_redundant = True
                    break
            if filter and not filter(s):
                continue
            elif is_redundant:
                continue
            else:
                summary_sents.append(sents[i])
                summary_vectors.append(new_x)

        summary = [s.raw for s in summary_sents]
        return summary


class CentroidRank(Summarizer):

    def __init__(self,  max_sim=0.9999):
        self.name = 'Sentence-Centroid Summarizer'
        self.max_sim = max_sim

    def score_sentences(self, X):
        Xsum = sparse.csr_matrix(X.sum(0))
        centroid = normalize(Xsum)
        scores = cosine_similarity(X, centroid)
        return scores

    def summarize(self, sents, k, vectorizer, filter=None, sbert=False):
        raw_sents = [s.raw for s in sents]
        if sbert:
            X = vectorizer.encode(
                    raw_sents,
                    batch_size=128,
                    show_progress_bar=True, 
                    device='cpu',
                    num_workers=24)
            for i, s in enumerate(sents):
                s.vector = X[i]
        else:
            try:
                X = vectorizer.transform(raw_sents)
                for i, s in enumerate(sents):
                    s.vector = X[i]
            except:
                return None
        

        scores = self.score_sentences(X)
        indices = list(range(len(sents)))
        ranked = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)

        summary_sents = []
        summary_vectors = []
        for i, _ in ranked:
            if len(summary_sents) >= k:
                break
            new_x = X[i]
            s = sents[i]
            is_redundant = False
            for x in summary_vectors:
                if cosine_similarity(new_x, x)[0, 0] > self.max_sim:
                    is_redundant = True
                    break
            if filter and not filter(s):
                continue
            elif is_redundant:
                continue
            else:
                summary_sents.append(sents[i])
                summary_vectors.append(new_x)

        summary = [s.raw for s in summary_sents]
        return summary


class CentroidOpt(Summarizer):

    def __init__(self, max_sim=0.9999):
        self.name = 'Summary-Centroid Summarizer'
        self.max_sim = max_sim

    def optimise(self, centroid, X, sents, k, filter):
        remaining = set(range(len(sents)))
        selected = []
        while len(remaining) > 0 and len(selected) < k:
            if len(selected) > 0:
                summary_vector = sparse.vstack([X[i] for i in selected])
                summary_vector = sparse.csr_matrix(summary_vector.sum(0))
            i_to_score = {}
            for i in remaining:
                if len(selected) > 0:
                    new_x = X[i]
                    new_summary_vector = sparse.vstack([new_x, summary_vector])
                    new_summary_vector = normalize(new_summary_vector.sum(0))
                else:
                    new_summary_vector = X[i]
                score = cosine_similarity(new_summary_vector, centroid)[0, 0]
                i_to_score[i] = score

            ranked = sorted(i_to_score.items(), key=lambda x: x[1], reverse=True)
            for i, score in ranked:
                s = sents[i]
                remaining.remove(i)
                if filter and not filter(s):
                    continue
                elif self.is_redundant(i, selected, X):
                    continue
                else:
                    selected.append(i)
                    break
        return selected

    def is_redundant(self, new_i, selected, X):
        summary_vectors = [X[i] for i in selected]
        new_x = X[new_i]
        for x in summary_vectors:
            if cosine_similarity(new_x, x)[0] > self.max_sim:
                return True
        return False

    def summarize(self, sents, k, vectorizer, filter=None, sbert=False):
        raw_sents = [s.raw for s in sents]
        try:
            X = vectorizer.transform(raw_sents)
        except:
            return None
        X = sparse.csr_matrix(X)
        Xsum = sparse.csr_matrix(X.sum(0))
        centroid = normalize(Xsum)
        selected = self.optimise(centroid, X, sents, k, filter)
        summary = [sents[i].raw for i in selected]
        return summary


class SubmodularSummarizer(Summarizer):
    """
    Selects a combination of sentences as a summary by greedily optimising
    a submodular function.
    The function date_models the coverage and diversity of the sentence combination.
    """
    def __init__(self, a=5, div_weight=6, cluster_factor=0.2):
        self.name = 'Submodular Summarizer'
        self.a = a
        self.div_weight = div_weight
        self.cluster_factor = cluster_factor

    def cluster_sentences(self, X):
        n = X.shape[0]
        n_clusters = round(self.cluster_factor * n)
        if n_clusters <= 1 or n <= 2:
            return dict((i, 1) for i in range(n))
        clusterer = MiniBatchKMeans(n_clusters=n_clusters)
        labels = clusterer.fit_predict(X)
        i_to_label = dict((i, l) for i, l in enumerate(labels))
        return i_to_label

    def compute_summary_coverage(self,
                                 alpha,
                                 summary_indices,
                                 sent_coverages,
                                 pairwise_sims):
        cov = 0
        for i, i_generic_cov in enumerate(sent_coverages):
            i_summary_cov = sum([pairwise_sims[i, j] for j in summary_indices])
            i_cov = min(i_summary_cov, alpha * i_generic_cov)
            cov += i_cov
        return cov

    def compute_summary_diversity(self,
                                  summary_indices,
                                  ix_to_label,
                                  avg_sent_sims):

        cluster_to_ixs = collections.defaultdict(list)
        for i in summary_indices:
            l = ix_to_label[i]
            cluster_to_ixs[l].append(i)
        div = 0
        for l, l_indices in cluster_to_ixs.items():
            cluster_score = sum([avg_sent_sims[i] for i in l_indices])
            cluster_score = np.sqrt(cluster_score)
            div += cluster_score
        return div

    def optimise(self,
                 sents,
                 k,
                 filter,
                 ix_to_label,
                 pairwise_sims,
                 sent_coverages,
                 avg_sent_sims):

        alpha = self.a / len(sents)
        remaining = set(range(len(sents)))
        selected = []

        while len(remaining) > 0 and len(selected) < k:

            i_to_score = {}
            for i in remaining:
                summary_indices = selected + [i]
                cov = self.compute_summary_coverage(
                    alpha, summary_indices, sent_coverages, pairwise_sims)
                div = self.compute_summary_diversity(
                    summary_indices, ix_to_label, avg_sent_sims)
                score = cov + self.div_weight * div
                i_to_score[i] = score

            ranked = sorted(i_to_score.items(), key=lambda x: x[1], reverse=True)
            for i, score in ranked:
                s = sents[i]
                remaining.remove(i)
                if filter and not filter(s):
                    continue
                else:
                    selected.append(i)
                    break

        return selected

    def summarize(self, sents, k, vectorizer, filter=None, sbert=False):
        raw_sents = [s.raw for s in sents]
        if sbert:
            X = vectorizer.encode(
                    raw_sents,
                    batch_size=128,
                    show_progress_bar=True, 
                    device='cpu',
                    num_workers=24)
        else:
            try:
                X = vectorizer.transform(raw_sents)
            except:
                return None

        ix_to_label = self.cluster_sentences(X)
        pairwise_sims = cosine_similarity(X)
        sent_coverages = pairwise_sims.sum(0)
        avg_sent_sims = sent_coverages / len(sents)

        selected = self.optimise(
            sents, k, filter, ix_to_label,
            pairwise_sims, sent_coverages, avg_sent_sims
        )

        summary = [sents[i].raw for i in selected]
        return summary

class SBERTSummarizer(Summarizer):
    def __init__(self, 
            sbert_model='distilroberta-base-paraphrase-v1',
            sbert_sequence_len=512,
            batch_size=128,
            device='cpu',
            date_only=False,
            summary_criteria='similarity',
            candidate_sents_per=5,
            compare_with='both',
            ):
        '''
        Args:
            sbert_model: sentence_transformers pretrained model to use
            sbert_sequence_len: int max length of tokens for sbert to use
            batch_size: for SBERT encoding
            device: for SBERT encoding. 'cpu', 'cuda', etc

            date_only: candidate sentences must be published on or mention date

            summary_criteria: 'similarity' or 'centroid'. Candidate sentences
                will be ranked by their cos similarity to the centroid of
                all candidate sentences and cluster documents (when 'centroid')
                or their mean similarity with all examples in the two groups
                (when 'similarity')
            candidate_sents_per: the first this number of sentences from each
                article published on a date that is the event date will be 
                considered target sentences
            compare_with: when not 'both', summary criteria only used for
                selected set of texts instead of using mean with both
        '''
        self.name = 'SBERT Summarizer'
        self.sbert_model = sbert_model
        self.sbert_sequence_len = sbert_sequence_len

        self.batch_size = batch_size
        self.device = device

        self.date_only = date_only

        self.sbert = SentenceTransformer(self.sbert_model)
        self.sbert.sbert_sequence_len = self.sbert_sequence_len - 3

        self.summary_criteria = summary_criteria
        self.candidate_sents_per = candidate_sents_per
        self.compare_with = compare_with

    def summarize(self, articles, k, date, filter=None):
        ''' 
        Generates summary with SBERT from sentence_transformers 
        
        Args:
            articles: list of articles to summarize
            k: summary length in number of sentences
            date: date of event articles represent

        Returns:
            list of raw sentences that make up summary (in order)
            [] if no valid summary
        '''
        if self.date_only:
            return self._summarize_date_only(articles, k, date)
        else:
            return self._summarize(articles, k, date)

        return

    def _summarize_date_only(self, articles, k, date):
        candidate_sents = []

        for a in articles:
            start_ind = 0

            if a.time.date() == date:
                start_ind = self.candidate_sents_per
                for s in a.sentences[:start_ind]:
                    candidate_sents.append(s)
            
            for s in a.sentences[start_ind:]:
                if s.get_date() and s.get_date() == date:
                    candidate_sents.append(s)

        if len(candidate_sents) == 0:
            return []

        print("\t...encoding candidate sentences...")
        candidate_sents_text = [s.raw for s in candidate_sents]
        candidate_sents_vecs = self.sbert.encode(
            candidate_sents_text,
            batch_size=self.batch_size,
            show_progress_bar=True, 
            device=self.device,
            num_workers=24)

        full_texts = ['{}. {}'.format(a.title, a.text) for a in articles]
        article_vecs = self.sbert.encode(
            full_texts,
            batch_size=self.batch_size,
            show_progress_bar=True, 
            device=self.device,
            num_workers=24)

        if self.summary_criteria == 'centroid':
            doc_compare_vecs = np.mean(article_vecs, axis=0)
            sent_compare_vecs = np.mean(candidate_sents_vecs, axis=0)
            assert len(doc_compare_vecs) == len(sent_compare_vecs)
        elif self.summary_criteria == 'similarity':
            doc_compare_vecs = article_vecs
            sent_compare_vecs = candidate_sents_vecs
        else:
            raise NotImplementedError

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

        top_sent_inds = np.argsort(-sent_scores)[:k]

        summary = []
        for ind in top_sent_inds:
            summary.append(candidate_sents_text[ind])

        return summary

    def _summarize(self, articles, k, date=None):
        candidate_sents = []

        candidate_sents = [
            s for a in articles for s in a.sentences[:self.candidate_sents_per]
        ]

        if len(candidate_sents) == 0:
            return []

        print("\n\t...encoding candidate sentences...")
        candidate_sents_text = [s.raw for s in candidate_sents]
        candidate_sents_vecs = self.sbert.encode(
            candidate_sents_text,
            batch_size=self.batch_size,
            show_progress_bar=True, 
            device=self.device,
            num_workers=24)

        print("\n\t...encoding cluster articles...")
        full_texts = ['{}. {}'.format(a.title, a.text) for a in articles]
        article_vecs = self.sbert.encode(
            full_texts,
            batch_size=self.batch_size,
            show_progress_bar=True, 
            device=self.device,
            num_workers=24)

        if self.summary_criteria == 'centroid':
            doc_compare_vecs = np.mean(article_vecs, axis=0)
            sent_compare_vecs = np.mean(candidate_sents_vecs, axis=0)
            assert len(doc_compare_vecs) == len(sent_compare_vecs)
        elif self.summary_criteria == 'similarity':
            doc_compare_vecs = article_vecs
            sent_compare_vecs = candidate_sents_vecs
        else:
            raise NotImplementedError

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

        top_sent_inds = np.argsort(-sent_scores)[:k]

        summary = []
        for ind in top_sent_inds:
            summary.append(candidate_sents_text[ind])

        return summary