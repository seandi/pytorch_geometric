from typing import Union, Tuple, List, Dict, Optional, Callable
from collections import defaultdict
from os import path
import datetime

from matplotlib import pyplot as plt
from matplotlib import cm, colors

import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap

import numpy as np

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


class DGM:
    r"""
    The Deep Graph Mapper module for graph visualization from the `"Deep Graph Mapper: Seeing Graphs through the
    Neural Lens" <https://arxiv.org/abs/2002.03864>`_ paper. This implementation is based on the authors' `official
    implementation <https://github.com/crisbodnar/dgm>`_.
    Given an input graph, DGM maps it to a new graph that compactly represents the original one using the Mapper
    algorithm combined with a given trainable lens function (refer to section 3.1 of the DGM
    `paper <https://arxiv.org/abs/2002.03864>`_ for an overview of Mapper). The mapped graph is further processed
    to filter out components that are too small and then is visualized.

    Args:
        lens_function (Data -> ndarray):
            The lens functions that takes as input the :obj:`Data` object representing the graph to visualize and
            produces the node features to be used for the visualization.
        num_intervals (int):
            The number of intervals over the :math:`[0,1]` interval to be used for generating the visualization.
            More intervals results in more fine-grained visualizations of the graph.
        overlap (float):
            The ratio of overlap between two consecutive intervals, expressed as a value between
            :math:`0` (disjoint intervals) and :math:`1` (fully overlapping).
            More overlap between intervals results in higher connectivity of the visualized graph.
        embedding_dims (int):
            The dimension of the node features/embeddings used for creating the visualized graph.
            If the lens function produces nodes embeddings with higher dimensions than this value,
            a dimensionality reduction algorithm is applied on top. Should be either :math:`1` or :math:`2`.
        embedding_reduction_method (str, optional):
            The dimensionality algorithm to be used if the lens function is high dimensional. (default: :obj:`"tsne"`)
        reduction_random_state (int, optional):
            The random initial state of the dimensionality reduction algorithm for reproducible results.
            (default: :obj:`0`).
        min_component_size (int, optional):
            The minimum size that a connected component of the mapped graph should have to not be
            filtered out in the visualized graph. The size of the component is given by the sum of
            the size of its nodes, each of which is given byb the number of nodes in the connected
            components of the original graph that has been mapped to that node. (default: :obj:`0`)
        use_sdgm (bool, optional):
            Flag to use Structural Deep Graph Mapper, instead of DGM. This changes how the edges of
            the mapped graph are constructed. See Appendix B of the DGM `paper <https://arxiv.org/abs/2002.03864>`_.
            (default: :obj:`False`)

    """

    _AVAILABLE_REDUCTION_METHODS = ['tsne', 'pca', 'isomap']

    def __init__(self,
                 lens_function: Callable[[Data], np.ndarray],
                 num_intervals: int, overlap: float, embedding_dims: int,
                 embedding_reduction_method: str = 'tsne',
                 reduction_random_state: Optional[int] = 0,
                 min_component_size: int = 0, use_sdgm: bool = False):

        assert embedding_dims == 1 or embedding_dims == 2, \
            f"Number of dimensions for the nodes embeddings is not valid," \
            f" {embedding_dims} was given but should be 1 or 2"
        assert 0.0 <= overlap <= 1.0, \
            f"Interval overlap percentage is not valid, got {overlap} but should be between 0 and 1"
        assert embedding_reduction_method in self._AVAILABLE_REDUCTION_METHODS, \
            f"{embedding_reduction_method} is not a valid reduction methods," \
            f" supported ones are {self._AVAILABLE_REDUCTION_METHODS} "

        self.lens_function = lens_function
        self.num_intervals = num_intervals
        self.overlap = overlap
        self.num_dims = embedding_dims
        self.reduction_method = embedding_reduction_method
        self.reduction_random_state = reduction_random_state
        self.min_comp_size = min_component_size
        self.use_sdgm = use_sdgm

        self.intervals: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]] = self._make_intervals()
        if self.num_dims == 2:
            self.intervals_x, self.intervals_y = self.intervals

    def _make_intervals(self, low: float = 0.0, high: float = 1.0
                        ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generates self.num_intervals equally sized intervals in the [0,1] range such that each two consecutive
        intervals have an overlap that is a self.overlap factor of the length of each interval.
        """

        # Find the interval length satisfying the constraints.
        x = (high - low) / (self.num_intervals - self.overlap * self.num_intervals + self.overlap)

        intervals_low = np.linspace(start=low, stop=high - x, num=self.num_intervals).reshape((-1, 1))
        intervals_high = np.linspace(start=low + x, stop=high, num=self.num_intervals).reshape((-1, 1))

        if self.num_dims == 1:
            intervals = np.concatenate((intervals_low, intervals_high), axis=1)
            return intervals

        if self.num_dims == 2:
            xx_low, yy_low = np.meshgrid(intervals_low, intervals_low, indexing='ij')
            xx_high, yy_high = np.meshgrid(intervals_high, intervals_high, indexing='ij')

            xx_low = xx_low.reshape((self.num_intervals ** 2, 1))
            yy_low = yy_low.reshape((self.num_intervals ** 2, 1))
            xx_high = xx_high.reshape((self.num_intervals ** 2, 1))
            yy_high = yy_high.reshape((self.num_intervals ** 2, 1))

            intervals_xx = np.concatenate((xx_low, xx_high), axis=1)
            intervals_yy = np.concatenate((yy_low, yy_high), axis=1)

            return intervals_xx, intervals_yy

    def make_visualization(self,
                           data: Data,
                           use_true_labels: bool = False,
                           figsize=(13, 11),
                           node_size_scale: Tuple[float, float] = (100.0, 1000.0),
                           edge_width_scale: Tuple[float, float] = (1.0, 15.0),
                           legend_dict: Optional[Dict] = None,
                           discrete_colormap: bool = False,
                           save_dir: Optional[str] = None,
                           name: Optional[str] = None):
        r"""

        Args:
            data (Data):
                The :obj:`Data` object representing the input graph.
            use_true_labels (bool, optional):
                Set this flag to :obj:`True` to use the nodes true labels (if available) to determine the color of
                the nodes in the visualization instead of using the (reduced) node embeddings given by the
                lens function. (default: :obj:`False`)
            figsize (Tuple[int,int], optional):
                The size of the visualization figure. (default: :obj:'(13, 11)')
            node_size_scale(Tuple[int, int], optional):
                The scale of node sizes. (default: :obj:`(100.0, 1000.0)`)
            edge_width_scale(Tuple[int, int], optional):
                The scale of edge widths. (default: :obj:`(1.0, 15.0)`)
            legend_dict (dict, optional):
                A :obj:`dict` associating to each node label its legend entry. Used only when using true labels.
                (default: :obj:`None`)
            discrete_colormap(bool, optional):
                Setting this to :obj:`True` will use a discrete colormap for the node colors, instead of the
                continuous colormap in the :math:`[0,1]` range. (default: :obj:`False`)
            save_dir(str, optional):
                Where to save the visualization image, if :obj:`None` the image is show instead. (deafault: :obj:`None`)
            name(str, optional):
                Custom file name for the visualization image. (deafault: :obj:`None`)
        """

        if use_true_labels:
            node_true_labels = data.y.detach().cpu().numpy()
        else:
            node_true_labels = None

        embeddings = self._compute_embeddings(data=data)

        pull_back_cover = self._compute_pull_back_cover(data=data, nodes_embeddings=embeddings)

        # Build the mapped graph from the pull back cover.
        mapped_graph = self._build_mapped_graph(data=data,
                                                node_embeddings=embeddings,
                                                pull_back_cover=pull_back_cover,
                                                node_true_labels=node_true_labels)

        filtered_graph = self._filter_small_components(mapped_graph=mapped_graph)

        self._plot_graph(mapped_graph=filtered_graph,
                         using_true_labels=False,
                         figsize=figsize,
                         node_size_scale=node_size_scale,
                         edge_width_scale=edge_width_scale,
                         legend_dict=legend_dict,
                         discrete_colormap=discrete_colormap,
                         save_dir=save_dir,
                         name=name)

        if use_true_labels:
            self._plot_graph(mapped_graph=filtered_graph,
                             using_true_labels=True,
                             figsize=figsize,
                             node_size_scale=node_size_scale,
                             edge_width_scale=edge_width_scale,
                             legend_dict=legend_dict,
                             discrete_colormap=discrete_colormap,
                             save_dir=save_dir,
                             name=name)

    def _compute_embeddings(self, data: Data) -> np.ndarray:
        """
        Compute the node embeddings f(X) as the normalized output of the lens function.
        A dimensionality reduction algorithm is used if the lens function is higher dimensional.
        Args:
            data(Data): The data for input graph.

        Returns:
            ndarray: the normalized embeddings
        """
        embeddings = self.lens_function(data)

        # Reduce embeddings dimensionality
        if embeddings.shape[1] > self.num_dims:
            print(f"Applying dimensionality reduction method ({self.reduction_method}) to the embeddings")
            if self.reduction_method == 'tsne':
                reduction = TSNE(n_components=self.num_dims, n_jobs=-1,
                                 random_state=self.reduction_random_state)
            elif self.reduction_method == 'pca':
                reduction = PCA(n_components=self.num_dims,
                                random_state=self.reduction_random_state)
            elif self.reduction_method == 'isomap':
                reduction = Isomap(n_components=self.num_dims, n_jobs=-1)
            else:
                raise NotImplementedError

            embeddings = reduction.fit_transform(embeddings)

        # Normalize the embeddings
        embeddings -= np.min(embeddings, axis=0, keepdims=True)
        embeddings /= (np.max(embeddings, axis=0, keepdims=True) + 1e-9)

        return embeddings

    def _compute_pull_back_cover(self, data: Data, nodes_embeddings: np.ndarray) -> List[np.ndarray]:
        r"""
        Computes the pull back cover f*(U) of the nodes of the input graph X with respect to the
        intervals of the open cover `U` and lens function f. Given the nodes embeddings
        f(X), for each interval U_i the subset of nodes X_i belonging to the preimage of
        f(U_i), f-1(U_i), is computed and the collection of those subsets forms the pull back cover
        f*(U)={f-1(U_i)}_i. See section 3.1 of https://arxiv.org/abs/2002.03864.

        Args:
            data (Data): The data for the input graph X.
            nodes_embeddings (ndarray): The embeddings f(X) of the graph nodes produced by the lens function.

        Returns:
            List[ndarray]: The pull back cover.

        """
        nodes_indices = np.arange(data.num_nodes)
        pull_back_cover = []

        for i in range(self.num_intervals ** self.num_dims):
            if self.num_dims == 1:
                mask = DGM._nodes_are_in_interval(interval=self.intervals[i, :], nodes_embeddings=nodes_embeddings)
                preimage_i = nodes_indices[mask]
            else:
                mask = np.logical_and(
                    DGM._nodes_are_in_interval(self.intervals_x[i, :], nodes_embeddings=nodes_embeddings[:, 0]),
                    DGM._nodes_are_in_interval(self.intervals_y[i, :], nodes_embeddings=nodes_embeddings[:, 1])
                )
                preimage_i = nodes_indices[mask]

            if preimage_i.shape[0] > 0:
                pull_back_cover.append(preimage_i)

        return pull_back_cover

    @staticmethod
    def _nodes_are_in_interval(interval: np.ndarray, nodes_embeddings: np.ndarray) -> np.ndarray:
        mask = np.logical_and(interval[0] <= nodes_embeddings, nodes_embeddings < interval[1]).squeeze()
        return mask

    def _build_mapped_graph(self, data: Data,
                            node_embeddings: np.ndarray,
                            pull_back_cover: List[np.ndarray],
                            node_true_labels: Optional[np.ndarray] = None
                            ) -> nx.Graph:
        """
        Builds the mapped graph from the node embeddings and the corresponding pull back cover.
        The mapped graph contains a node for each connected component in each subset in the pull back cover.
        Edges are added to the mapped graph depending on the interconnectivity of those components.
        Args:
            data (Data): The graph data
            node_embeddings (ndarray): The (normalized) embeddings.
            pull_back_cover (List[ndarray]): Ther pull back cover
            node_true_labels (bool):
             If ture use the node true labels for determining the mapped graph node colors, instead of the embeddings.

        Returns:
            Graph: the mapped graph.
        """

        original_graph: nx.Graph = to_networkx(data, to_undirected=True)
        mapped_graph = nx.Graph()

        mapped_graph_num_nodes: int = 0

        # Assign to each node in the mapped graph the list of original nodes mapped to it
        mapped_node_to_original_nodes: Dict[int, List[int]] = {}
        # Assign to each original node the list of mapped nodes it has been mapped to
        original_node_to_mapped_nodes: Dict[int, List[int]] = {}
        mapped_nodes_color: Dict[int, Union[int, np.ndarray]] = {}
        mapped_nodes_color_true_labels: Dict[int, Union[int, np.ndarray]] = {}
        mapped_nodes_size: Dict[int, int] = {}

        # Add a node to the mapped graph for each connected component in each of the pull back cover sets
        for preimage_nodes in pull_back_cover:
            subgraph = original_graph.subgraph(nodes=preimage_nodes)
            conn_components = nx.connected_components(subgraph)

            for conn_comp_nodes in conn_components:
                conn_comp_nodes_list = list(conn_comp_nodes)

                # Update nodes mapping with the new node
                mapped_node_to_original_nodes[mapped_graph_num_nodes] = conn_comp_nodes_list

                for node in conn_comp_nodes_list:
                    if node not in original_node_to_mapped_nodes.keys():
                        original_node_to_mapped_nodes[node] = []
                    original_node_to_mapped_nodes[node].append(mapped_graph_num_nodes)

                # Compute the feature(s) of the new node defining its color in the visualization
                conn_comp_embeddings = node_embeddings[list(conn_comp_nodes)]
                conn_comp_color = np.mean(conn_comp_embeddings, axis=0)

                mapped_nodes_color[mapped_graph_num_nodes] = conn_comp_color

                # If true labels are available, compute also a coloring based on them
                if node_true_labels is not None:
                    conn_comp_predictions = node_true_labels[list(conn_comp_nodes)]
                    labels, freq = np.unique(conn_comp_predictions, return_counts=True)
                    mapped_node_label = labels[np.argmax(freq)]

                    mapped_nodes_color_true_labels[mapped_graph_num_nodes] = mapped_node_label

                # Compute the size of the new node
                mapped_nodes_size[mapped_graph_num_nodes] = len(conn_comp_nodes_list)
                mapped_graph_num_nodes += 1

        # Insert all the new nodes in the mapped graph
        mapped_graph.add_nodes_from(np.arange(mapped_graph_num_nodes))
        nx.set_node_attributes(mapped_graph, values=mapped_nodes_color, name='color')
        if node_true_labels is not None:
            nx.set_node_attributes(mapped_graph, values=mapped_nodes_color_true_labels, name='color_true_labels')
        nx.set_node_attributes(mapped_graph, values=mapped_nodes_size, name='size')

        # Compute the edges of the mapped graph
        mapped_graph_edges: Dict[Tuple[int, int], int] = defaultdict(lambda: 0)

        if self.use_sdgm:
            # If an edge in the original graph connects two endpoints mapped to different nodes,
            # add this edge top the mapped graph
            for edge in original_graph.edges:
                u, v = edge[0], edge[1]
                for mapped_node1 in original_node_to_mapped_nodes[u]:
                    for mapped_node2 in original_node_to_mapped_nodes[v]:
                        if mapped_node1 != mapped_node2:
                            mapped_graph_edges[min(mapped_node1, mapped_node2), max(mapped_node1, mapped_node2)] += 1
        else:
            # If a node is mapped to multiple connected components then connect all of them in the mapped graph
            for node in original_node_to_mapped_nodes.keys():
                for i, mapped_node1 in enumerate(original_node_to_mapped_nodes[node]):
                    for mapped_node2 in original_node_to_mapped_nodes[node][i + 1:]:
                        mapped_graph_edges[min(mapped_node1, mapped_node2), max(mapped_node1, mapped_node2)] += 1

        # Compute the normalized edge weights
        edge_weight = np.fromiter(mapped_graph_edges.values(), dtype=float)
        edge_weight = edge_weight / np.max(edge_weight)

        # Add the edges to the mapped graph
        for i, edge in enumerate(mapped_graph_edges.keys()):
            mapped_graph.add_edge(u_of_edge=edge[0], v_of_edge=edge[1], weight=edge_weight[i])

        print("Mapped graph nodes", mapped_graph.number_of_nodes())
        print("Mapped graph edges", mapped_graph.number_of_edges())

        return mapped_graph

    def _filter_small_components(self, mapped_graph: nx.Graph) -> nx.Graph:
        """
        Filters out from the mapped graph the connected components whose size is smaller than a threshold,
        then prunes dangling edges. The size S_m(v) of a node v in the mapped graph is given by the number of nodes
        contained in the connected component of the original graph that has been mapped to node v. The size of a
        connected component in the mapped graph is the sum its nodes sizes.

        Args:
            mapped_graph (Graph): the mapped graph.

        Returns:
            Graph: the filtered mapped graph.
        """
        filtered_graph = nx.Graph()

        # Boolean mask to filter out small nodes
        nodes_mask = np.zeros(shape=(mapped_graph.number_of_nodes(),), dtype=bool)
        kept_nodes = []

        # Recover nodes size (i.e. size of the corresponding conn comp in the original graph)
        node_size = np.fromiter(nx.get_node_attributes(mapped_graph, name='size').values(), dtype=float)

        # Determine which components in the mapped graph are to be kept according to the cumulative size of the
        # connected components in the original graph associated to the nodes of the conn comp in the mapped graph
        for conn_comp in nx.connected_components(mapped_graph):
            comp_size = np.sum([node_size[node] for node in conn_comp])
            if comp_size >= self.min_comp_size:
                nodes_mask[np.array(list(conn_comp))] = True
                kept_nodes.extend(list(conn_comp))

        # Add filtered nodes to the graph
        filtered_nodes = np.arange(nodes_mask.sum())
        filtered_graph.add_nodes_from(filtered_nodes)

        # Recover nodes color
        node_color = np.array(list(nx.get_node_attributes(mapped_graph, name='color').values()))
        filtered_node_color = node_color[nodes_mask]
        nx.set_node_attributes(filtered_graph,
                               values=dict(zip(filtered_nodes, filtered_node_color)),
                               name='color')

        # Recover coloring with true labels if available
        node_color_true_labels = list(nx.get_node_attributes(mapped_graph, name='color_true_labels').values())
        if len(node_color_true_labels) == node_color.shape[0]:
            filtered_node_color_true_labels = np.array(node_color_true_labels)[nodes_mask]
            nx.set_node_attributes(filtered_graph,
                                   values=dict(zip(filtered_nodes, filtered_node_color_true_labels)),
                                   name='color_true_labels')


        # Recover nodes size
        filtered_node_size = node_size[nodes_mask]
        nx.set_node_attributes(filtered_graph,
                               values=dict(zip(filtered_nodes, filtered_node_size)),
                               name='size')

        # Create the map associating the index of each kept node in the mapped graph to its index in the filtered graph
        kept_nodes.sort()
        map_to_filtered_node = dict(zip(kept_nodes, filtered_nodes))

        # Add edges to the filtered graph
        for edge in mapped_graph.edges():
            if nodes_mask[edge[0]] and nodes_mask[edge[1]]:
                filtered_graph.add_edge(u_of_edge=map_to_filtered_node[edge[0]],
                                        v_of_edge=map_to_filtered_node[edge[1]],
                                        weight=mapped_graph.get_edge_data(*edge)['weight']
                                        )

        print("Filtered graph nodes", filtered_graph.number_of_nodes())
        print("Filtered graph edges", filtered_graph.number_of_edges())

        return filtered_graph

    def _plot_graph(self,
                    mapped_graph: nx.Graph,
                    figsize=(13, 11),
                    using_true_labels: bool = False,
                    node_size_scale: Tuple[float, float] = (100.0, 1000.0),
                    edge_width_scale: Tuple[float, float] = (1.0, 15.0),
                    legend_dict: Optional[Dict] = None,
                    discrete_colormap: bool = False,
                    save_dir: Optional[str] = None,
                    name: Optional[str] = None
                    ) -> None:

        assert not discrete_colormap or (self.num_dims == 1 or legend_dict is not None), \
            f"Discrete colormap not supported when using embeddings with more than one dimension"

        # Retrieve and rescale nodes size
        node_size = np.fromiter(nx.get_node_attributes(mapped_graph, name='size').values(), dtype=float)
        node_size /= np.max(node_size)
        node_size = node_size * (node_size_scale[1] - node_size_scale[0]) + node_size_scale[0]

        # Retrieve and rescale edge weights
        edge_weight = np.fromiter(nx.get_edge_attributes(mapped_graph, name='weight').values(), dtype=float)
        edge_weight /= np.max(edge_weight)
        edge_width = edge_weight * (edge_width_scale[1] - edge_width_scale[0]) + edge_width_scale[0]

        # Retrieve nodes color
        if using_true_labels:
            node_color = list(nx.get_node_attributes(mapped_graph, name='color_true_labels').values())
            assert len(node_color) == node_size.shape[0]
            node_color = np.array(node_color)
        else:
            node_color = np.array(list(nx.get_node_attributes(mapped_graph, name='color').values()))

        if self.num_dims == 2 and not using_true_labels:
            # Compute bivariate colormap
            node_color = DGM._make_2d_color_map(values_x=node_color[:, 0], values_y=node_color[:, 1])
            xx, yy = np.mgrid[0:100, 0:100]
            bivariate_colormap = DGM._make_2d_color_map(xx, yy)

        # Create figure
        figure: plt.Figure = plt.figure(figsize=figsize)
        axis = figure.add_subplot()
        axis.axis("off")

        # Load colormap
        if using_true_labels or discrete_colormap:
            if np.max(node_color) >= 7:
                cmap = cm.get_cmap('tab20')
            else:
                cmap = cm.get_cmap('Dark2')
        else:
            cmap = cm.get_cmap('coolwarm')
        plt.set_cmap(cmap)

        # Compute nodes visualization layout
        pos = nx.nx_pydot.graphviz_layout(mapped_graph)

        # Draw the nodes
        if self.num_dims == 2 and not using_true_labels:
            # Draw graph on the left
            ax_graph = axis.inset_axes([.0, .0, .79, 1.0])
            nx.draw(mapped_graph, pos=pos, node_color=node_color,
                    node_size=node_size, width=edge_width,
                    ax=ax_graph, cmap=cmap)

            # Plot bivariate colormap reference as a image on the right
            axin = axis.inset_axes([0.8, 0.4, .19, .19])
            axin.imshow(bivariate_colormap, extent=[0.0, 1.0, 0.0, 1.0])

        else:
            nx.draw(mapped_graph, pos=pos, node_color=node_color.ravel(),
                    node_size=node_size, width=edge_width,
                    ax=axis, cmap=cmap)

        # Plot colorbar/legend
        if using_true_labels or discrete_colormap:
            color_norm = colors.Normalize(vmin=0, vmax=np.max(node_color))
            scalar_map = cm.ScalarMappable(norm=color_norm, cmap=cmap)

            # plot legend instead of colorbar if available
            if legend_dict is not None:
                for label in legend_dict:
                    axis.plot([0], [0], color=scalar_map.to_rgba(label), label=legend_dict[label])

                figure.legend()
            else:
                figure.colorbar(scalar_map, shrink=0.5)
        elif self.num_dims == 1:
            # Plot continuous colorbar wrt the embeddings scale
            scalar_map = cm.ScalarMappable(cmap=cmap)
            figure.colorbar(scalar_map, shrink=0.5)

        if save_dir is not None:
            if name is None:
                name = "plot_" + str(datetime.datetime.now()).replace(" ", "-").replace(":", "-").split(".")[0]
            figure.tight_layout()
            file_path = path.join(save_dir, name + ".png")
            plt.savefig(file_path, dgi=300)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def _make_2d_color_map(values_x: np.ndarray, values_y: np.ndarray):
        values_x = np.array(
            (values_x - values_x.min()) / (values_x.max() - values_x.min()) * 255, dtype=np.int32)
        values_y = np.array(
            (values_y - values_y.min()) / (values_y.max() - values_y.min()) * 255, dtype=np.int32)

        cmap_x = cm.get_cmap('cool')
        cmap_y = cm.get_cmap('coolwarm')

        color_x = cmap_x(values_x)
        color_y = cmap_y(values_y)

        color_xy_map = np.sum([color_x, color_y], axis=0) / 2.0

        return color_xy_map
