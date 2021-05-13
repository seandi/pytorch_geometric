from typing import Tuple

import numpy as np

import torch
from torch import nn, Tensor
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_adj, dense_to_sparse

import scipy
from scipy import sparse


class MPRPooling(nn.Module):
    r"""The Mapping-Based PageRank (MPR) pooling layer from `"Deep Graph Mapper: Seeing Graphs through the Neural Lens"
    <https://arxiv.org/abs/2002.03864>`_ (section 4.3.2). This implementation is based on the authors'
    `official implementation <https://github.com/crisbodnar/dgm.git>`_.

    The MPR pooling layer first creates a set of :obj:`n` equal length overlapping intervals in the :math:`[0,1]`
    range and applies the `PageRank` algorithm to assign to each node in the input graph a score :math:`pr_i`.
    The pooled graph is generated by creating one node for each sub-interval :math:`[l,r]` whose features are obtained
    by pooling together the features of all nodes in the input graph having a pagerank score within the :math:`[l,r]`
    interval. Two nodes in the pooled graph are connected by an edge if the corresponding sub-graphs in the original
    graph shares a common edge.

    Args:
        out_nodes (int):
            The number of nodes in the pooled output graph.
        overlap (float):
            The overlap ratio defined as the ratio between the
            length of the overlap between two consecutive sub-intervals and
            the length of each sub-interval.
        eps (float):
            Small value used to provide numerical stability
            when normalizing the pagerank scores (dafault: :math:`10^{-9}`).
    """
    def __init__(self, out_nodes: int, overlap: float, eps: float = 1e-9):
        super(MPRPooling, self).__init__()

        self.eps = eps
        self.out_nodes = out_nodes
        self.intervals = self._make_1d_intervals()

    def forward(self, x: Tensor, edge_index: Adj) -> Tuple[Tensor, Adj]:
        r""" Given the input graph, computes the pooled output graph using MPR pooling.

        Args:
            x (Tensor):
                The node features.
            edge_index (Adj):
                The edge indices.

        Returns:
             (:class:`Tensor`, :class:`Adj`): The node features and edge index of the pooled graph.

        """
        # Convert sparse edges representation to dense adjacency matrix for PageRank
        adj = to_dense_adj(edge_index=edge_index)[0]
        in_nodes = x.shape[0]
        nodes_index = np.arange(in_nodes)

        # Init selection matrix with dimension (nodes_in_pooled_graph, nodes_in_original_graph)
        S = torch.zeros((self.out_nodes, in_nodes))

        pagerank = self._compute_pagerank(adj)

        # Compute selection matrix S
        for out_node, interval in enumerate(self.intervals):
            # compute set of nodes to pool together according to their pagerank
            nodes_to_pool = nodes_index[np.logical_and(pagerank >= interval[0], pagerank < interval[1])]
            # Skip if there are no nodes to pool for this interval
            if nodes_to_pool.shape[0] > 0:
                S[out_node, nodes_to_pool] = 1.0

        # Normalize selection matrix
        S = torch.softmax(S, dim=0).to(x.device)

        # Generate the pooled graph
        x = S @ x
        adj = S @ adj @ S.t()
        edge_index, _ = dense_to_sparse(adj)

        return x, edge_index

    def _compute_pagerank(self, adj: Tensor) -> np.ndarray:
        r"""Computes the pagerank of the graph using a sparse implementation of the PageRank algorithm

        Args:
            adj (Tensor): The adjacency matrix of the graph.

        Returns:
            ndarray: The vector of pagerank scores of the nodes in the input graph.

        """
        adj = adj.detach().cpu().numpy()

        pagerank_result = MPRPooling._sparse_pagerank_power(adj)

        # Scale the pagerank values in the unit interval
        pagerank_result -= np.min(pagerank_result)
        pagerank_result /= max(np.max(pagerank_result), self.eps)
        pagerank_result = np.ravel(pagerank_result)

        return pagerank_result

    @staticmethod
    def _sparse_pagerank_power(
            adj: np.ndarray, alpha: float = 0.85,
            max_iter: int = 100, tol: float = 1e-6
    ) -> np.ndarray:
        r"""Implementation of the iterative PageRank algorithm using sparse matrix based on "Experiments with MATLAB,
        Chapter 7" https://it.mathworks.com/moler/exm.html and https://github.com/asajadi/fast-pagerank.git.

        Args:
            adj (ndarray):
                The adjacency matrix of the graph.
            alpha (float, optional):
                alpha coefficient of the PageRank algorithm.
            max_iter (int, optional):
                The maximum number of iterations.
            tol (float, optional):
                The tolerance threshold.

        Returns:
            ndarray: The vector of pagerank scores of the nodes in the input graph.
        """
        n = adj.shape[0]

        A_sparse = sparse.csr_matrix(adj.transpose())
        I = sparse.eye(n)
        ones = np.ones((n,1))

        c = np.asarray(A_sparse.sum(axis=0)).reshape(-1)
        k = c.nonzero()[0]
        D = sparse.csr_matrix((1 / c[k], (k, k)), shape=(n, n))

        z_T = (((1 - alpha) * (c != 0) + (c == 0)) / n)[np.newaxis, :]
        W = alpha * A_sparse @ D

        x = ones
        oldx = np.zeros((n, 1))

        iteration = 0

        while scipy.linalg.norm(x - oldx) > tol:
            oldx = x
            x = W @ x + ones @ (z_T @ x)
            iteration += 1
            if iteration >= max_iter:
                break
        x = x / sum(x)

        return x

    def _make_1d_intervals(self, low: float = 0.0, high: float = 1.0) -> np.ndarray:
        """
        Generates the (n_intervals, 2) matrix with the lower and upper bounds of each sub-interval.
        """
        assert 0.0 <= self.overlap <= 1.0, f"The overlap percentage should be a value between 0.0 and 1.0"

        # Compute the length x of the intervals s.t. each two consecutive intervals
        # have an overlap of length dx=overlap*x
        x = (high - low) / (self.intervals - self.overlap * self.intervals + self.overlap)

        intervals_low = np.linspace(start=low, stop=high - x, num=self.intervals).reshape((-1, 1))
        intervals_high = np.linspace(start=low + x, stop=high, num=self.intervals).reshape((-1, 1))
        intervals = np.concatenate((intervals_low, intervals_high), axis=1)

        return intervals