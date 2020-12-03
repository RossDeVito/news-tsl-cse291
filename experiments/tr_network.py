import datetime
from pprint import pprint
import collections

import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer

from sknetwork.topology import CoreDecomposition
from sknetwork.clustering import Louvain, KMeans, PropagationClustering
from sknetwork.hierarchy import LouvainHierarchy
from sknetwork.ranking import PageRank

from news_tls import utils, data, summarizers



class NetworkTimelineGenerator():
	def __init__(self,
				 cluster_method='datewise',
				 cluster_n_mult=2,
				 clip_sents=5,
				 use_pub_date=True,
				 doc_to_date_binary=True,
				 entity_max_df=.5,
				 correlation_p_val=.05,
				 kcore=2,
				 article_selection='adj',
				 summarizer=None,
				 unique_dates=True):
		'''
		Constructs text-rich network based timeline generator

		Args:
			cluster_method: algorithm used to cluster nodes in heterogeneous 
				network. Can be 'kmeans'
					'datewise' - finds most important dates in whole network,
						then finds associated articles with each date
					'kmeans' - finds clusters where k = max_dates on timeline
						then finds most important date in cluster and associated
						articles
			cluster_n_mult: for kmeans, generate this * max_dates clusters and
				max_dates top valid clusters
			clip_sents: int or None. If int, first n sentences from each article
				used. If None, whole articles used
			use_pub_date: when True articles always have an edge to their
				publication date. When False, only if date mentioned in text
				as determined by Tilse
			doc_to_date_binary: if doc to date edges should be binary. Are tfidf
				when false
			entity_max_df: float or int. Entities with document frequency
				greater than this are ignored
			correlation_p_val: correlation (date to date or entity to entity)
				used as edge weight (instead of 0 for no edge) when correlation
				is positive and p value is less than correlation_p_val.
			kcore: int or None. Constructed network will be decomposed to k-core
				before clustering when int. When None, full network used
			article_selection: method for selecting articles from selected date.
				'adj' - selects all articles adjacent to date
			summarizer: summarizer from news_tls/summarizers.py. Defaults to 
				summarizers.CentroidOpt
			unique_dates: True if timeline can have at most one event per day
		'''

		self.cluster_method = cluster_method
		self.cluster_n_mult = cluster_n_mult
		self.clip_sents = clip_sents
		self.use_pub_date = use_pub_date
		self.doc_to_date_binary = doc_to_date_binary
		self.entity_max_df = entity_max_df
		self.correlation_p_val = correlation_p_val
		self.kcore = kcore

		self.article_selection = article_selection

		self.summarizer = summarizer or summarizers.CentroidOpt()
		self.unique_dates = unique_dates

	def predict(self,
				collection,
				max_dates=10,
				max_summary_sents=1,
				ref_tl=None,
				input_titles=False,
				output_titles=False,
				output_body_sents=True):

		print('construsting network...')
		network = self._build_network_from_collection(collection)

		if self.kcore:
			print('decomposing to {}-core network...'.format(self.kcore))
			network = self._kcore_decompose(network)

		# datewise approach
		if self.cluster_method == 'datewise':
			print("Finding most important dates...")
			pagerank = PageRank()

			full_pr = pagerank.fit_transform(network['adj'])
			sorted_nodes = np.argsort(-full_pr)

			sorted_dates = []
			sorted_date_inds = []

			for i in sorted_nodes:
				if network['types'][i] == 'date':
					sorted_dates.append(network['values'][i])
					sorted_date_inds.append(i)
					if len(sorted_dates) >= max_dates * 2:
						break

			timeline = []
			for d in sorted_dates:
				timeline.append((datetime.datetime(d.year, d.month, d.day), ''))
			timeline.sort(key=lambda x: x[0])

			print('finding associated articles...')
			sorted_clusters = []
			if self.article_selection == 'adj':
				for date, ind in zip(sorted_dates, sorted_date_inds):
					adj_inds = network['adj'][ind].indices
					adj_types = network['types'][adj_inds]

					sorted_clusters.append(
						[
							date,
							network['values'][adj_inds[adj_types == 'doc']]
						]
					)
			else:
				raise NotImplementedError()
		# clustering approaches
		elif self.cluster_method == 'kmeans':
			print("clustering...")
			kmeans = KMeans(n_clusters=round(max_dates * self.cluster_n_mult))
			labels = kmeans.fit_transform(network['adj'])

			cluster_graphs = []

			for clust in np.unique(labels):
				cluster_mask = labels == clust
				cluster_graphs.append(
					{
						'adj': network['adj'][np.ix_(cluster_mask, cluster_mask)],
						'values': network['values'][cluster_mask],
						'types': network['types'][cluster_mask]
					}
				)

				if not self._is_valid_cluster(cluster_graphs[-1]):
					print("@@@@@@@@@ INVALID CLUSTER @@@@@@@@@")

		if self.cluster_method in ['kmeans']:
			print('finding cluster date and articles...')
			sorted_clusters = []

			for net in cluster_graphs:
				pagerank = PageRank()

				full_pr = pagerank.fit_transform(net['adj'])
				sorted_nodes = np.argsort(-full_pr)

				most_important_date = None
				for ind in sorted_nodes:
					if net['types'][ind] == 'date':
						most_important_date = (net['values'][ind], ind)
						break

				if most_important_date is None:
					print("\tcluster has no most important date")
					continue

				if most_important_date and (self.article_selection == 'adj'):
					adj_inds = net['adj'][most_important_date[1]].indices
					adj_types = net['types'][adj_inds]

					sorted_clusters.append(
						[
							most_important_date[0],
							net['values'][adj_inds[adj_types == 'doc']]
						]
					)
				else:
					raise NotImplementedError()

		print('fitting vectorizer...')
		raw_sents = [s.raw for a in collection.articles() for s in
					 a.sentences[:self.clip_sents]]
		vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
		vectorizer.fit(raw_sents)

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

		for d, articles in sorted_clusters:
			print("\t{}\t{} articles".format(d, len(articles)))
			date = d
			c_sents = self._select_sents_from_articles(articles)
			print("\t\t{} sents".format(len(c_sents)))
			summary = self.summarizer.summarize(
				c_sents,
				k=max_summary_sents,
				vectorizer=vectorizer,
				filter=sent_filter
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

	def _is_valid_cluster(self, network, require=['doc', 'date', 'entity']):
		''' returns if cluster has required types '''
		for req_type in require:
			if req_type not in network['types']:
				print("{} type not in network".format(req_type))
				return False
		return True

	def _build_network_from_collection(self, collection):
		''' 
		Builds network from collection of articles 
		
		returns:
			network as a dictionary with:
				'adj': adjacency sparse matrix
				'values': node values running down axis
				'types: node types down axis
		'''

		# which entities to use as nodes, see https://spacy.io/api/annotation
		node_entities = [
			'PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT',
			'WORK_OF_ART', 'LAW', 'LANGUAGE'
		]

		doc_id = []		# now contains article itself
		doc_dates = []
		doc_entities = []

		for doc in collection.articles():
			doc_id.append(doc)

			if self.clip_sents:
				sents = doc.sentences[:self.clip_sents]
			else:
				sents = doc.sentences				

			dates = [sent.get_date() for sent in sents if sent.get_date()]
			if self.use_pub_date:
				dates.append(doc.time.date())
			doc_dates.append(dates)

			ents = [
				ent[0].lower() for s in sents for ent in s.get_entities() 
					if ent[1] in node_entities
			]
			doc_entities.append(ents)

		# Vectorize docs to get edge weights
		date_tfidf_vectorizer = TfidfVectorizer(
			binary=self.doc_to_date_binary,
			lowercase=False,
			analyzer='word',
			tokenizer=lambda x: x,
			preprocessor=lambda x: x,
			token_pattern=None)

		date_tfidf = date_tfidf_vectorizer.fit_transform(doc_dates)
		date_feat_names = date_tfidf_vectorizer.get_feature_names()

		ent_tfidf_vectorizer = TfidfVectorizer(
			max_df=self.entity_max_df,
			analyzer='word',
			tokenizer=lambda x: x,
			preprocessor=lambda x: x,
			token_pattern=None)

		ent_tfidf = ent_tfidf_vectorizer.fit_transform(doc_entities)
		ent_feat_names = ent_tfidf_vectorizer.get_feature_names()

		# use doc reps to get date and entity correlations
		date_corrs, p_vals = spearmanr(date_tfidf.A)
		date_corrs[(date_corrs < 0) | (p_vals > self.correlation_p_val)] = 0
		np.fill_diagonal(date_corrs, 0)

		ent_corrs, p_vals = spearmanr(ent_tfidf.A)
		ent_corrs[(ent_corrs < 0) | (p_vals > self.correlation_p_val)] = 0
		np.fill_diagonal(ent_corrs, 0)

		# get node values and types
		node_values = np.asarray(doc_id + date_feat_names + ent_feat_names)
		node_types = np.asarray((
			['doc'] * len(doc_id) 
			+ ['date'] * len(date_feat_names) 
			+ ['entity'] * len(ent_feat_names)
		))

		# build adjacency matrix with tfidf edges and correlations
		adjacency = np.zeros((len(node_values), len(node_values)))

		num_docs = len(doc_id)
		num_dates = len(date_feat_names)
		num_not_ent = num_docs + num_dates

		adjacency[:num_docs, num_docs:num_docs+num_dates] = date_tfidf.A
		adjacency[num_docs:num_docs+num_dates, :num_docs] = date_tfidf.T.A

		adjacency[:num_docs, num_not_ent:] = ent_tfidf.A
		adjacency[num_not_ent:, :num_docs] = ent_tfidf.T.A

		adjacency[num_docs:num_docs+num_dates, num_docs:num_docs+num_dates] = date_corrs
		adjacency[num_not_ent:, num_not_ent:] = ent_corrs

		adjacency = sparse.csr_matrix(adjacency)

		return {
			'adj': adjacency, 
			'values': node_values, 
			'types': node_types
		}

	def _kcore_decompose(self, network):
		''' return k-core decomposed network '''
		kcore = CoreDecomposition()
		core_val = kcore.fit_transform(network['adj'])
		
		core_mask = core_val >= self.kcore
		adjacency = network['adj'][np.ix_(core_mask, core_mask)]
		values = network['values'][core_mask]
		types = network['types'][core_mask]

		n_nodes = len(network['values'])
		new_n_nodes = len(values)
		print("\t {} / {} nodes remain ({:.3%})".format(
				new_n_nodes, n_nodes, (new_n_nodes / n_nodes)))

		return {
			'adj': adjacency, 
			'values': values, 
			'types': types
		}

	
	def _select_sents_from_articles(self, articles):
		sents = []
		for a in articles:
			pub_d = a.time.date()
			for s in a.sentences[:self.clip_sents]:
				sents.append(s)
		return sents

	def load(self, ignored_topics):
		''' Does not need to load pretrained model '''
		pass


def is_valid_cluster(graph, require=['doc', 'date', 'entity']):
	for req_type in require:
		if req_type not in graph['types']:
			print("{} type not in graph".format(req_type))
			return False
	return True

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


if __name__ == "__main__":
	''' For testing and dev '''
	from pathlib import Path
	from news_tls import data
	from tilse.data.timelines import Timeline as TilseTimeline
	from tilse.data.timelines import GroundTruth as TilseGroundTruth
	import itertools


	dataset_path = Path("../../data/entities")
	trunc_timelines = True
	time_span_extension = 7

	if not dataset_path.exists():
		raise FileNotFoundError(f'Dataset not found: {dataset_path}')
	dataset = data.Dataset(dataset_path)
	dataset_name = dataset_path.name

	# from inside evaluate, where this would eventually go
	results = []
	metric = 'align_date_content_costs_many_to_one'
	# evaluator = rouge.TimelineRougeEvaluator(measures=["rouge_1", "rouge_2"])
	n_topics = len(dataset.collections)

	# just use first topic and timeline for dev
	collection = dataset.collections[0]

	# get reference timeline and its l and k
	ref_timelines = [
		TilseTimeline(tl.date_to_summaries) for tl in collection.timelines
	]

	ref_timeline = data.truncate_timelines(ref_timelines, collection)[0]
	ref_dates = sorted(ref_timeline.dates_to_summaries)

	start, end = data.get_input_time_span(ref_dates, time_span_extension)

	collection.start = start
	collection.end = end

	l = len(ref_dates)
	k = data.get_average_summary_length(ref_timeline)

	ground_truth = TilseGroundTruth([ref_timeline])

	# a10 = list(itertools.islice(collection.articles(), 10))

	# by whole article or clip
	whole_article = False
	clip_sents = 5
	use_pub_date = True

	# which entities to use as nodes, see https://spacy.io/api/annotation
	node_entities = [
		'PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT',
		'WORK_OF_ART', 'LAW', 'LANGUAGE'
	]

	doc_id = []
	doc_dates = []
	doc_entities = []

	for doc in tqdm(collection.articles()):
		doc_id.append(doc.id)

		if whole_article:
			sents = doc.sentences			
		else:
			sents = doc.sentences[:clip_sents]

		dates = [sent.get_date() for sent in sents if sent.get_date()]
		if use_pub_date:
			dates.append(doc.time.date())
		doc_dates.append(dates)

		ents = [
			ent[0].lower() for s in sents for ent in s.get_entities() 
				if ent[1] in node_entities
		]
		doc_entities.append(ents)

	# Vectorize docs to get edge weights
	dates_binary_edges = True

	date_tfidf_vectorizer = TfidfVectorizer(
		analyzer='word',
		binary=dates_binary_edges,
		tokenizer=lambda x: x,
		preprocessor=lambda x: x,
		lowercase=False,
		token_pattern=None)

	date_tfidf = date_tfidf_vectorizer.fit_transform(doc_dates)
	date_feat_names = date_tfidf_vectorizer.get_feature_names()
	
	ent_max_df = .5

	ent_tfidf_vectorizer = TfidfVectorizer(
		max_df=ent_max_df,
		analyzer='word',
		tokenizer=lambda x: x,
		preprocessor=lambda x: x,
		token_pattern=None)

	ent_tfidf = ent_tfidf_vectorizer.fit_transform(doc_entities)
	ent_feat_names = ent_tfidf_vectorizer.get_feature_names()

	# use doc reps to get date and entity correlations
	## will only use positive corr w/ p val < .05
	date_corrs, p_vals = spearmanr(date_tfidf.A)
	date_corrs[(date_corrs < 0) | (p_vals > .05)] = 0
	np.fill_diagonal(date_corrs, 0)

	# sns.distplot(date_corrs[date_corrs > 0].ravel(), rug=True, kde=False)

	ent_corrs, p_vals = spearmanr(ent_tfidf.A)
	ent_corrs[(ent_corrs < 0) | (p_vals > .05)] = 0
	np.fill_diagonal(ent_corrs, 0)

	combined_feats = np.hstack([date_tfidf.A, ent_tfidf.A])

	feat_corrs, p_vals = spearmanr(combined_feats)
	feat_corrs[(feat_corrs < 0) | (p_vals > .05)] = 0
	np.fill_diagonal(feat_corrs, 0)

	# add node names and type
	node_names = np.asarray(doc_id + date_feat_names + ent_feat_names)
	node_types = np.asarray((
		['doc'] * len(doc_id) 
		+ ['date'] * len(date_feat_names) 
		+ ['entity'] * len(ent_feat_names)
	))

	# build adjacency matrix with tfidf edges and correlations
	adjacency = np.zeros((len(node_names), len(node_names)))

	num_docs = len(doc_id)
	num_dates = len(date_feat_names)
	num_not_ent = num_docs + num_dates

	adjacency[:num_docs, num_docs:num_docs+num_dates] = date_tfidf.A
	adjacency[num_docs:num_docs+num_dates, :num_docs] = date_tfidf.T.A

	adjacency[:num_docs, num_not_ent:] = ent_tfidf.A
	adjacency[num_not_ent:, :num_docs] = ent_tfidf.T.A

	adjacency[num_docs:num_docs+num_dates, num_docs:num_docs+num_dates] = date_corrs
	adjacency[num_not_ent:, num_not_ent:] = ent_corrs

	adjacency = sparse.csr_matrix(adjacency)

	# get k core, want only globally relevant nodes
	kcore_val = 2
	if kcore_val:
		print("Decomposing to {}-core".format(kcore_val))

		kcore = CoreDecomposition()
		core_val = kcore.fit_transform(adjacency)
		
		core_mask = core_val >= kcore_val
		kcore_adjacency = adjacency[np.ix_(core_mask, core_mask)]
		kcore_types = node_types[core_mask]

		if is_valid_cluster({'adj': kcore_adjacency, 'types':kcore_types}):
			n_nodes = len(node_names)

			adjacency = kcore_adjacency
			node_types = kcore_types
			node_names = node_names[core_mask]

			new_n_nodes = len(node_names)
			print("\t {} / {} nodes remain ({:.3%})".format(
				new_n_nodes, n_nodes, (new_n_nodes / n_nodes))
			)

		else:
			print("kcore decoposition results in invalid graph")

	# cluster 
	print("Clustering")
	kmeans = KMeans(n_clusters=l)
	labels = kmeans.fit_transform(adjacency)

	pagerank = PageRank()
	pr_scores = pagerank.fit_transform(adjacency)

	cluster_graphs = []

	for clust in np.unique(labels):
		cluster_mask = labels == clust
		cluster_graphs.append(
			{
				'adj': adjacency[np.ix_(cluster_mask, cluster_mask)],
				'names': node_names[cluster_mask],
				'types': node_types[cluster_mask]
			}
		)

		if not is_valid_cluster(cluster_graphs[-1]):
			print("@@@@@@@@@ INVALID CLUSTER @@@@@@@@@")


	# denoise and select date


	# get most important dates at different levels and compare
	print("Ref dates:")
	pprint(ref_dates)

	pagerank = PageRank()

	print("\nFull graph pagerank:")
	full_pr = pagerank.fit_transform(adjacency)
	sorted_nodes = np.argsort(-full_pr)
	sorted_dates = []
	for i in sorted_nodes:
		if node_types[i] == 'date':
			sorted_dates.append(node_names[i])
			if len(sorted_dates) >= l:
				break
	pprint(sorted_dates)

	timeline = []
	for d in sorted_dates:
		timeline.append((datetime.datetime(d.year, d.month, d.day), ''))
	timeline.sort(key=lambda x: x[0])

	fgpr_tl = TilseTimeline(data.Timeline(timeline).date_to_summaries)

	date_scores = evaluate_dates(fgpr_tl, ground_truth)
	pprint(date_scores)

	

	






	

	# # generate timelines with other models
	# model_timelines = {}
	# model_scores = {}

	# # add ground truth
	# rouge_scores = get_scores(
	# 		metric, ref_timeline, ground_truth, evaluator)
	# date_scores = evaluate_dates(ref_timeline, ground_truth)
	# wordmover_score = get_wordmover_score(ref_timeline, ground_truth)
	# supert_score = get_supert_score(
	# 		list(collection.articles())[:],
	# 		ref_timeline,
	# 		ref_timeline)

	# model_timelines['ground truth'] = ref_timeline
	# model_scores['ground truth'] = [
	# 	rouge_scores, date_scores, wordmover_score, supert_score
	# ]

	# # run other models on timeline
	# for model, model_name in tls_models:
	# 	print("Predicting with {}".format(model_name))

	# 	model.load(ignored_topics=[collection.name])

	# 	# generate timeline
	# 	model_timelines[model_name] = TilseTimeline(
	# 		model.predict(
	# 			collection,
	# 			max_dates=l,
	# 			max_summary_sents=k,
	# 			ref_tl=ref_timeline # only oracles need this
	# 		).date_to_summaries
	# 	)

	# 	# evaluate timeline
	# 	rouge_scores = get_scores(
	# 		metric, model_timelines[model_name], ground_truth, evaluator)
	# 	date_scores = evaluate_dates(model_timelines[model_name], ground_truth)
	# 	wm_score = get_wordmover_score(model_timelines[model_name], ground_truth)
	# 	supert_score = get_supert_score(
	# 		list(collection.articles())[:],
	# 		model_timelines[model_name],
	# 		ref_timeline)
		
	# 	model_scores[model_name] = [
	# 		rouge_scores, date_scores, wm_score, supert_score
	# 	]