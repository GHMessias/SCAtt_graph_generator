import sys
sys.path.append('')

from core.attributed_graph import AttributedGraph
from core.base_graph_generator import BaseGenerator
import numpy as np
import networkx as nx
from tqdm.auto import tqdm
import logging

logging.basicConfig(level=logging.INFO)

def theoretical_intra_class_estimates(c_probs, native_probs, m, approx_limit=100000):
    '''
    Estimate the theoretical (in-the-limit) intra-class edge probabilities. Has some variance due to the stochasticity in empirical power-law generation and theoretical probabilities from the underlying BA model.
    
    Note: Supports 3 variants of c_probs:
    -Callable (degree-dependent). Ex: c_probs = lambda k: {1: np.tanh(k/5), 0: 1 - np.tanh(k/5)}
    -Precomputed (degree-dependent) dictionary.  Ex: c_probs = {k: {1: np.tanh(k/5), 0: 1 - np.tanh(k/5)} for k in range(100)}
    -Fixed (constant).  Ex: c_probs = {1: p_c, 0: 1 - p_c}
    
    `approx_limit` is used to approximate the infinite sums in Eqs. 5-6 from the paper.
    '''
    
    g = np.sum(np.power(native_probs, 2))  # probability of an intra-class edge to arise given certain class propensities
    if callable(c_probs):
        # Dynamically computed, degree dependent
        within = np.sum([(g * c_probs(k)[1] * (m+1)) / ((k+1) * (k+2)) for k in range(m, approx_limit)])
        cross = np.sum([((1-g) * (1-c_probs(k)[1]) * (m+1)) / ((k+1) * (k+2)) for k in range(m, approx_limit)])
    else:
        if len(c_probs) == 2:
            # Fixed
            within = np.sum([(g * c_probs[1] * (m+1)) / ((k+1) * (k+2)) for k in range(m, approx_limit)])
            cross = np.sum([((1-g) * (1-c_probs[1]) * (m+1)) / ((k+1) * (k+2)) for k in range(m, approx_limit)])
        else:
            # Precomputed, degree-dependent
            within = np.sum([(g * c_probs[k][1] * (m+1)) / ((k+1) * (k+2)) for k in range(m, approx_limit)])
            cross = np.sum([((1-g) * (1-c_probs[k][1]) * (m+1)) / ((k+1) * (k+2)) for k in range(m, approx_limit)])
    
    intra_class_ratio = within / (within + cross)
    return intra_class_ratio


def draw_network_with_labels(G, node_labels):
    '''
    Given a networkx graph and labels, draw nodes with relevant colors.
    
    G: networkx graph on N nodes
    node_labels: label vector of length N (assuming K labels, this would be labeled 0...K-1)
    
    Note: color_mapping currently fixed to maximum 5 colors, but can trivially be extended.
    '''
    
    color_mapping = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'purple'}
    node_colors = [color_mapping[i] for i in node_labels]
    nx.draw_kamada_kawai(G, with_labels=False, node_size=50, node_color=node_colors)


def cabam_graph_generation(n, m, c=2, native_probs=[0.5, 0.5], c_probs={1: 0.5, 0: 0.5}, logger=None):
    '''
    Main function for CABAM graph generation.
    
    n: maximum number of nodes
    m: number of edges to add at each timestep (also the minimum degree)
    c: number of classes
    native_probs: c-length vector of native class probabilities (must sum to 1)
    c_probs: p_c from the paper.  Entry for 1 (0) is the intra-class (inter-class) link strength.  Entries must sum to 1.
    
    Supports 3 variants of c_probs:
    -Callable (degree-dependent). Ex: c_probs = lambda k: {1: np.tanh(k/5), 0: 1 - np.tanh(k/5)}
    -Precomputed (degree-dependent) dictionary.  Ex: c_probs = {k: {1: np.tanh(k/5), 0: 1 - np.tanh(k/5)} for k in range(100)}
    -Fixed (constant).  Ex: c_probs = {1: p_c, 0: 1 - p_c}
    '''
    
    if m < 1 or n < m:
        raise nx.NetworkXError(
                "NetworkXError must have m>1 and m<n, m=%d,n=%d" % (m, n))

    logger = logger if logger else logging
    
    # graph initialization
    G = nx.empty_graph(m)
    intra_class_edges = 0
    inter_class_edges = 0
    total_intra_class = 0
    total_inter_class = 0
    
    intra_class_ratio_tracker = []
    alpha_tracker = []

    class_tbl = list(range(c))
    node_labels = np.array([np.random.choice(class_tbl, p=native_probs) for x in range(G.number_of_nodes())])
    node_degrees = np.array([1] * m) # technically degree 0, but using 1 here to make the math work out.

    # start adding nodes
    source = m
    source_label = np.random.choice(class_tbl, p=native_probs)
    pbar = tqdm(total=n)
    pbar.update(m)
    empirical_edge_fraction_to_degree_k = np.zeros(10)
    n_added = 0
    
    while source < n:
        logger.debug('Adding node {} with label {}.'.format(source, source_label))
        if type(c_probs) == dict:
            if len(c_probs) == 2:
                # no funny business, just constants
                node_class_probs = np.array([c_probs[abs(node_labels[i] == source_label)] for i in range(len(node_labels))])
            else:
                # pre-generated custom probabilities
                node_class_probs = np.array([c_probs[node_degrees[i]][abs(node_labels[i] == source_label)] for i in range(len(node_labels))])
        else:
            # callable (function) probs
            node_class_probs = np.array([c_probs(node_degrees[i])[abs(node_labels[i] == source_label)] for i in range(len(node_labels))])
            
        
        # determine m target nodes to connect to
        targets = []
        while len(targets) != m: 
            node_class_degree_probs = node_class_probs * node_degrees
            candidate_targets = np.where(node_class_degree_probs > 0)[0]

            if len(candidate_targets) >= m:
                logger.debug('Have enough targets...sampling from AWPA.\n')
                # if we have enough qualifying nodes, sample from assortativity-weighted PA probs
                candidate_node_class_degree_probs = node_class_degree_probs[candidate_targets]
                candidate_node_class_degree_probs = candidate_node_class_degree_probs / np.linalg.norm(node_class_degree_probs, ord=1)
                targets = np.random.choice(candidate_targets, 
                                           size=m, 
                                           p=candidate_node_class_degree_probs, 
                                           replace=False)
            else:
                logger.debug('Not enough targets...sampling from PA.\n')
                # else, use as many qualifying nodes as possible, and just sample from the PA probs for the rest.
                n_remaining_targets = m - len(candidate_targets)
                other_choices = np.where(node_class_degree_probs == 0)[0]
                other_node_degree_probs = node_degrees[other_choices]
                other_node_degree_probs = other_node_degree_probs / np.linalg.norm(other_node_degree_probs, ord=1)
                other_targets = np.random.choice(other_choices, 
                                                 size=n_remaining_targets, 
                                                 p=other_node_degree_probs,
                                                 replace=False)
                #print(candidate_targets, candidate_targets.shape, other_targets, other_targets.shape)
                targets = np.concatenate((candidate_targets, other_targets))
            assert len(targets) == m

        G.add_edges_from([(source, target) for target in targets])
        edge_types = np.array([source_label == node_labels[target] for target in targets])
        intra_class_edges += np.count_nonzero(edge_types) # intra-class edges
        inter_class_edges += np.count_nonzero(edge_types == 0) # inter-class edges
        
        total_intra_class += np.count_nonzero(edge_types)
        total_inter_class +=  np.count_nonzero(edge_types == 0)
        total_intra_frac = total_intra_class / (total_intra_class + total_inter_class)
        
        #intra_class_ratio_tracker.append(total_intra_frac)
        #alpha_tracker.append(test_degree_distribution(node_degrees)[0])
        
        ncdp = node_class_degree_probs / np.linalg.norm(node_class_degree_probs, ord=1)
        empirical_edge_fraction_to_degree_k += np.array([m*np.sum(ncdp[node_degrees == k]) for k in range(m, m+10)])
        
        if source % 500 == 0:
            theoretical_edge_fraction_to_degree_k = [((m*(m+1))/((k+1)*(k+2))) for k in range(m, m+10)]
            ncdp = node_class_degree_probs / np.linalg.norm(node_class_degree_probs, ord=1)
            avgd_empirical_edge_fraction_to_degree_k = empirical_edge_fraction_to_degree_k / n_added
            
            logger.info('Theor. edge prob to deg k: {}'.format(np.round(theoretical_edge_fraction_to_degree_k, 3)))
            logger.info('Empir. edge prob to deg k: {}'.format(np.round(avgd_empirical_edge_fraction_to_degree_k, 3)))
            snapshot_intra_frac = intra_class_edges / (intra_class_edges + inter_class_edges)
            logger.info('Snapshot: ({}/{})={:.3f}\t Overall: {:.3f}'.format(intra_class_edges, intra_class_edges+inter_class_edges,
                                                                             snapshot_intra_frac, total_intra_frac))
            intra_class_edges = 0
            inter_class_edges = 0
            logger.info('Max node degree: {}'.format(max(node_degrees)))

        # book-keeping
        node_degrees[targets] += 1
        node_labels = np.append(node_labels, source_label)
        node_degrees = np.append(node_degrees, m)
        pbar.update(1)

        # move onto next node!
        n_added += 1
        source += 1
        source_label = np.random.choice(class_tbl, p=native_probs)
    
    pbar.close()
    return G, node_degrees, node_labels, total_intra_class, total_inter_class, intra_class_ratio_tracker, alpha_tracker

import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
import networkx as nx
import pickle


def make_undirected_and_self_looped(A):
    '''takes scipy sparse matrix'''
    A_hat = A.maximum(A.T)  # make symmetric
    A_hat.setdiag(1)  # populate diagonal with self-loops
    return A_hat


def normalize_adjacency(A, style='left'):
    '''takes scipy sparse matrix'''
    d = np.asarray(A.sum(axis=1)).flatten().astype(float)  # degree vector
    if style == 'left':
        return (np.diag(d**-1.0)) @ A  # D^-1 * A
    return np.diag(d**-0.5) @ A @ np.diag(d**-0.5)  # D^0.5 * A * D^0.5


def normalize_features(H, p=1, style='row'):
    '''takes scipy sparse matrix where rows are node features'''
    if style == 'row':
        return H / (norm(H, ord=p, axis=1).reshape(H.shape[0], 1))
    else:
        return H / (norm(H, ord=p, axis=0).reshape(1, H.shape[1]))


def tensorize(*args, tensor_type):
    '''cast input arrays to torch tensors'''
    return [tensor_type(x) for x in args]


def get_train_val_test_masks(y, train_ratio, val_ratio, test_ratio):
    '''get indices for train, val and test sets given ratios'''
    n = len(y)
    ratio_arr = np.array([0, train_ratio, val_ratio, test_ratio]) / (train_ratio + val_ratio + test_ratio)
    split_idx = np.cumsum(ratio_arr * n, dtype=int)
    mask = np.ones(n)
    for mask_id, (st, end) in enumerate(zip(split_idx, split_idx[1:])):
        mask[st:end] = mask_id
    np.random.shuffle(mask)
    idx_train, idx_val, idx_test = [np.where(mask == mask_id)[0] for mask_id in [0, 1, 2]]
    return idx_train, idx_val, idx_test


def encode_onehot(y):
    '''produce onehot encoding of label vector y given a flattened class vector y'''
    n_samples = len(y)
    n_classes = len(y.unique())
    y_onehot = torch.LongTensor(n_samples, n_classes)
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    return y_onehot


def load_simulated_random_data():
    '''simulate some random data with random binary labels'''
    A = np.random.binomial(n=1, p=0.2, size=(50,50))
    H = np.random.rand(50, 128)
    y = np.random.binomial(n=1, p=0.2, size=(50,))
    
    # preprocess adjacency and features 
    A = make_undirected_and_self_looped(A)
    A = normalize_adjacency(A)
    H = normalize_features(H)
    
    idx_train, idx_val, idx_test = get_train_val_test_masks(y, 0.7, 0.1, 0.2)
    
    A, H = tensorize(A, H, tensor_type=torch.FloatTensor)
    y, idx_train, idx_val, idx_test = tensorize(y, idx_train, idx_val, idx_test, tensor_type=torch.LongTensor)
    
    return A, H, y, idx_train, idx_val, idx_test 


def load_simulated_clustered_data(n_clusters, n_samples_per_cluster, n_dims):
    '''simulate clustered multivariate gaussian data with adjacency stochastically sampled based on distance in R^d'''
    H = []
    y = []
    for i in range(n_clusters):
        H.append(np.random.multivariate_normal(mean=np.random.randint(10, size=n_dims), 
                                                      cov = np.eye(n_dims), 
                                                      size=n_samples_per_cluster))
        y.extend([i] * n_samples_per_cluster)
    H = np.vstack(H)
    y = np.array(y)
    
    # preprocess adjacency and features 
    A = make_undirected_and_self_looped(A)
    A = normalize_adjacency(A)
    H = normalize_features(H)
    
    A = cdist(H, H, metric='euclidean')
    A = 1.0 - (A - A.min(axis=1)) / (A.max(axis=1) - A.min(axis=1))
    A = np.random.binomial(n=1, p=A)
    idx_train, idx_val, idx_test = get_train_val_test_masks(y, 0.7, 0.1, 0.2)
    
    A, H = tensorize(A, H, tensor_type=torch.FloatTensor)
    y, idx_train, idx_val, idx_test = tensorize(y, idx_train, idx_val, idx_test, tensor_type=torch.LongTensor)
    
    return A, H, y, idx_train, idx_val, idx_test


def accuracy(output, labels):
    '''get accuracy given output tensor and labels '''
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def load_dataset(path, train_ratio, test_ratio, val_ratio):
    '''Load a graph from a Numpy binary file.
    Adapted from https://github.com/abojchevski/graph2gauss/blob/master/g2g/utils.py
    '''
    
    if not path.endswith('.npz'):
        path += '.npz'
    with np.load(path, allow_pickle=True) as loader:
        loader = dict(loader)
        A = csr_matrix((loader['adj_data'], loader['adj_indices'],
                        loader['adj_indptr']), shape=loader['adj_shape'])

        H = csr_matrix((loader['attr_data'], loader['attr_indices'],
                        loader['attr_indptr']), shape=loader['attr_shape'])

        y = loader.get('labels')
        
        # preprocess adjacency and features 
        A = make_undirected_and_self_looped(A)
        #A = normalize_adjacency(A)
        #H = normalize_features(H)
        return A, H, y
        
#         idx_train, idx_val, idx_test = get_train_val_test_masks(y, train_ratio, val_ratio, test_ratio)
        
#         A, H = tensorize(A, H, tensor_type=torch.FloatTensor)
#         y, idx_train, idx_val, idx_test = tensorize(y, idx_train, idx_val, idx_test, tensor_type=torch.LongTensor) 
        
#         return A, H, y, idx_train, idx_val, idx_test
#         graph = {'A': A, 'H': H, 'y': y}

#         idx_to_node = loader.get('idx_to_node')
#         if idx_to_node:
#             idx_to_node = idx_to_node.tolist()
#             graph['idx_to_node'] = idx_to_node

#         idx_to_attr = loader.get('idx_to_attr')
#         if idx_to_attr:
#             idx_to_attr = idx_to_attr.tolist()
#             graph['idx_to_attr'] = idx_to_attr

#         idx_to_class = loader.get('idx_to_class')
#         if idx_to_class:
#             idx_to_class = idx_to_class.tolist()
#             graph['idx_to_class'] = idx_to_class

#         return graph


def produce_processed_data(dataset):
    print('Loading dataset {}'.format(dataset))
    if dataset in ['airport', 'flickr', 'blogcatalog']:
        with open('data/alternative/{}_features.pkl'.format(dataset), 'rb') as f_obj, open('data/alternative/{}_adj.pkl'.format(dataset), 'rb') as a_obj, open('data/alternative/{}_labels.pkl'.format(dataset), 'rb') as l_obj:
                features = pickle.load(f_obj)
                adj = pickle.load(a_obj)
                labels = pickle.load(l_obj)

                if sp.issparse(features):
                    H = features.toarray()
                else:
                    H = features.numpy()

                if sp.issparse(adj):
                    A = adj.toarray()
                else:
                    A = adj

                if type(labels) != np.ndarray:
                    y = labels.numpy().ravel()
                else:
                    y = labels.ravel()
    else:
        A, H, y = load_dataset('data/{}/raw/{}.npz'.format(dataset, dataset), 
                               train_ratio=0.1, val_ratio=0.1, test_ratio=0.8)
        A = A.toarray()
        H = H.toarray()
    
    H[np.isnan(H)] = 0
    G = nx.from_numpy_matrix(A)
    
    return A, H, y, G




##############################
###### CABAM Generator #######
##############################

from torch_geometric.utils import from_networkx, to_networkit

class CABAMGenerator(BaseGenerator):
    '''
    Base para a execução do algoritmo CaBaM.
    '''
    def __init__(self):
        super().__init__(name = 'CaBaM', supports_mimic = False, supports_augment = False)

    def _run_cabam(self, num_nodes, num_connected_nodes, num_classes, native_probs, c_probs):
        G, node_degrees, node_labels, intra_class, inter_class, ratio_tracker, alpha_tracker = cabam_graph_generation(n = num_nodes, m = num_connected_nodes, native_probs=native_probs, c_probs=c_probs, c = num_classes)

        n_classes_in_G = len(np.unique(node_labels))

        feature_means = np.eye(n_classes_in_G)
        node_features = []
        for n_id, c_id in zip(range(len(G)), node_labels):
            f = np.random.multivariate_normal(mean=feature_means[c_id, :],
                                            cov=feature_means,
                                            size=1)
            node_features.append(f.ravel())

        node_features = np.array(node_features)

        x = torch.tensor(node_features)
        y = torch.tensor(node_labels)
        graph = to_networkit(edge_index = from_networkx(G).edge_index, directed = False)

        return AttributedGraph(graph = graph, x = x, y =y)

    
    def generate(self, num_nodes, num_connected_nodes, native_probs, c_probs, num_classes):
        return self._run_cabam(num_nodes = num_nodes, num_connected_nodes = num_connected_nodes, native_probs=native_probs, c_probs=c_probs, num_classes=num_classes)