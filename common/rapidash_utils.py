# MIT License, (c) 2024  (c) 2024 Sharvaree Vadgama
import torch

try:
    from torch_cluster import knn_graph, knn, fps, radius_graph as torch_radius_graph

    def fps_edge_index(pos, batch, ratio):
        fps_index = fps(pos, batch, ratio)

        pos_fps = pos[fps_index]
        batch_fps = batch[fps_index]

        edge_index = knn(pos_fps, pos, 1, batch_fps, batch)

        return edge_index, pos_fps, batch_fps

    def radius_graph(x, r, batch=None, loop=False, max_num_neighbors=32, flow="source_to_target"):
        """Computes the radius graph for points in x."""
        assert flow in ["source_to_target", "target_to_source"]
        edge_index = torch_radius_graph(x, r, batch, loop, max_num_neighbors=max_num_neighbors)
        
        if flow == "target_to_source":
            edge_index = edge_index.flip(0)
            
        return edge_index

except ModuleNotFoundError:
    import warnings

    warnings.warn(
        "Module `torch_cluster` not found. Defaulting to local clustering algorithms which may increase computation time."
    )

    def fps_edge_index(pos, batch, ratio):
        batch_size = batch.max().item() + 1
        all_target_nodes = []
        all_fps_indices = []

        num_samples_running = 0
        for b in range(batch_size):
            pos_b = pos[batch == b]
            num_nodes = pos_b.size(0)
            num_samples = max(int(ratio * num_nodes), 1)

            fps_indices = torch.zeros(num_samples, dtype=torch.long, device=pos.device)
            distances = torch.full((num_nodes,), float("inf"), device=pos.device)

            initial_index = torch.randint(0, num_nodes, (1,))
            fps_indices[0] = initial_index

            for i in range(1, num_samples):
                new_point = pos_b[fps_indices[i - 1]]
                current_distances = torch.norm(pos_b - new_point.unsqueeze(0), dim=1)
                distances = torch.min(distances, current_distances)
                fps_indices[i] = torch.argmax(distances)
            all_fps_indices.append(
                fps_indices + (batch == b).nonzero(as_tuple=True)[0].min()
            )

            # Compute edge_index: each source connected to nearest fps point
            dist_matrix = torch.cdist(pos_b, pos_b[fps_indices])
            nearest_indices = torch.argmin(dist_matrix, dim=1)
            target_nodes = nearest_indices + num_samples_running

            all_target_nodes.append(target_nodes)
            num_samples_running += num_samples

        source_nodes = torch.arange(pos.size(0), device=pos.device, dtype=torch.long)
        target_nodes = torch.cat(all_target_nodes)
        edge_index = torch.stack([source_nodes, target_nodes], dim=0)

        fps_indices = torch.cat(all_fps_indices)
        fps_pos = pos[fps_indices]
        fps_batch = batch[fps_indices]

        return edge_index, fps_pos, fps_batch

    def radius_graph(x, r, batch=None, loop=False, max_num_neighbors=32, flow="source_to_target"):
        """
        Computes the radius graph for points in x without torch_cluster.
        For each point, finds all neighbors within radius r.
        
        Args:
            x: Point cloud tensor of shape [N, D]
            r: Radius within which to connect points
            batch: Optional batch vector
            loop: If True, include self-loops
            max_num_neighbors: Maximum number of neighbors per node
            flow: Direction of edges ("source_to_target" or "target_to_source")
            
        Returns:
            edge_index: Graph connectivity in COO format with shape [2, E]
        """
        assert flow in ["source_to_target", "target_to_source"]
        
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
            
        edge_indices = []
        
        for b in torch.unique(batch):
            mask = batch == b
            x_batch = x[mask]
            
            # Compute pairwise distances
            dist_matrix = torch.cdist(x_batch, x_batch)
            
            # Find neighbors within radius
            neighbors = (dist_matrix <= r)
            if not loop:
                neighbors.fill_diagonal_(False)
                
            # Get source and target indices
            source_idx, target_idx = torch.where(neighbors)
            
            # Limit number of neighbors if necessary
            if max_num_neighbors is not None and max_num_neighbors > 0:
                # For each source node, randomly select up to max_num_neighbors if exceeded
                for i in range(x_batch.size(0)):
                    mask = source_idx == i
                    if mask.sum() > max_num_neighbors:
                        to_keep = torch.randperm(mask.sum())[:max_num_neighbors]
                        source_idx = torch.cat([source_idx[~mask], source_idx[mask][to_keep]])
                        target_idx = torch.cat([target_idx[~mask], target_idx[mask][to_keep]])
            
            # Offset indices based on batch
            offset = mask.nonzero(as_tuple=True)[0].min()
            source_idx += offset
            target_idx += offset
            
            edge_indices.append(torch.stack([source_idx, target_idx], dim=0))
            
        edge_index = torch.cat(edge_indices, dim=1)
        
        if flow == "target_to_source":
            edge_index = edge_index.flip(0)
            
        return edge_index

    def knn(x, y, k, batch_x=None, batch_y=None):
        """For every point in `y`, returns `k` nearest neighbors in `x`."""

        edge_index = y.new_empty(2, k * y.shape[0], dtype=torch.long)

        if batch_x is None:
            batch_x = x.new_zeros(x.shape[0], dtype=torch.long)

        if batch_y is None:
            batch_y = y.new_zeros(y.shape[0], dtype=torch.long)

        num_seen = 0

        for i, (b, b_size) in enumerate(
            zip(*torch.unique(batch_y, return_counts=True))
        ):
            x_b, y_b = x[batch_x == b], y[batch_y == b]

            batch_offset = i * b_size
            num_per_batch = k * b_size

            source = (
                torch.arange(b_size, device=b_size.device, dtype=torch.long)
            ).repeat_interleave(k) + batch_offset

            target = (
                torch.topk(torch.cdist(y_b, x_b), k, largest=False)[1].flatten()
                + batch_offset
            )

            edge_index[0, num_seen : num_seen + num_per_batch] = target
            edge_index[1, num_seen : num_seen + num_per_batch] = source

            num_seen += num_per_batch

        return edge_index

    def knn_graph(x, k, batch=None, loop=False, flow="source_to_target"):
        """
        For each point in `x`, calculates its `k` nearest neighbors.
        If `loop` is `True`, neighbors include self-connections.
        """
        assert flow in ["source_to_target", "target_to_source"]

        k += not loop

        edge_index = knn(x, x, k, batch, batch)

        if not loop:
            edge_index = edge_index[:, edge_index[0] != edge_index[1]]

        if flow == "target_to_source":
            edge_index = edge_index.flip(0)

        return edge_index


def fully_connected_edge_index(batch_idx):
    edge_indices = []

    for batch_num in torch.unique(batch_idx):
        # Find indices of nodes in the current batch
        node_indices = torch.where(batch_idx == batch_num)[0]
        grid = torch.meshgrid(node_indices, node_indices, indexing="ij")
        edge_indices.append(
            torch.stack([grid[0].reshape(-1), grid[1].reshape(-1)], dim=0)
        )

    edge_index = torch.cat(edge_indices, dim=1)

    return edge_index


def scatter_add(src, index, dim_size):
    out_shape = [dim_size] + list(src.shape[1:])
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    dims_to_add = src.dim() - index.dim()
    for _ in range(dims_to_add):
        index = index.unsqueeze(-1)
    index_expanded = index.expand_as(src)
    return out.scatter_add_(0, index_expanded, src)


def scatter_softmax(src, index, dim_size):
    src_exp = torch.exp(src - src.max())
    sum_exp = scatter_add(src_exp, index, dim_size) + 1e-6
    return src_exp / sum_exp[index]