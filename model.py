"""
Message-Passing GNNs from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - edges_to_coo
def edges_to_coo(edge_list, num_nodes=None):
    # TODO: Convert a list of (src, dst) edge pairs into COO-format src/dst tensors.
    if not isinstance(edge_list, torch.Tensor):
        edge_list = torch.tensor(edge_list, dtype=torch.long).reshape(-1, 2)

    src, dst = edge_list[:, 0], edge_list[:, 1]

    if num_nodes is None:
        if len(edge_list) == 0:
            num_nodes = 0
        else:
            num_nodes = int(torch.max(edge_list).item())+1

    return src, dst, num_nodes

# Step 2 - add_self_loops
def add_self_loops(src, dst, num_nodes):
    """Append self-loop edges (i, i) for every node to COO edge indices.

    Args:
        src: LongTensor [E] source node indices.
        dst: LongTensor [E] destination node indices.
        num_nodes: int, number of nodes in the graph.

    Returns:
        src_out: LongTensor [E + num_nodes]
        dst_out: LongTensor [E + num_nodes]
    """
    # TODO: Append self-loop edges (i, i) for every node to the COO tensors
    index = torch.arange(num_nodes, dtype=src.dtype, device=src.device)
    src_out = torch.cat([src, index], dim=0)
    dst_out = torch.cat([dst, index], dim=0)
    return src_out, dst_out

# Step 3 - compute_node_degrees
def compute_node_degrees(src, dst, num_nodes, edge_weight=None):
    """Compute per-node in-degrees (optionally weighted) from COO edges.

    Args:
        src (LongTensor): Source node indices of shape [E].
        dst (LongTensor): Destination node indices of shape [E].
        num_nodes (int): Number of nodes N.
        edge_weight (FloatTensor, optional): Per-edge weights of shape [E].

    Returns:
        FloatTensor: In-degrees of shape [N].
    """
    # TODO: Compute per-node in-degrees by scattering onto destination nodes
    if edge_weight is None:
        edge_weight = torch.ones(src.shape[0], dtype=torch.float32)
    degrees = torch.zeros(num_nodes, dtype=torch.float32)
    return degrees.scatter_add_(0, dst, edge_weight)

# Step 4 - symmetric_normalize_edge_weights
def symmetric_normalize_edge_weights(src, dst, num_nodes, edge_weight=None):
    """Compute symmetrically normalized edge weights w_ij / sqrt(d_i * d_j).

    Args:
        src (LongTensor): Source node indices of shape [E].
        dst (LongTensor): Destination node indices of shape [E].
        num_nodes (int): Number of nodes N.
        edge_weight (FloatTensor, optional): Per-edge weights of shape [E].
            Defaults to all ones (float32) when None.

    Returns:
        FloatTensor: Symmetrically normalized weights of shape [E].
    """
    # TODO: Compute symmetrically normalized edge weights for GCN-style propagation.
    if edge_weight is None:
        edge_weight = torch.ones(src.shape[0], dtype=torch.float32)
    deg = compute_node_degrees(src, dst, num_nodes, edge_weight)
    
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0

    # Compute normalized weights: w_ij / sqrt(d_i * d_j)
    norm = deg_inv_sqrt[src] * edge_weight * deg_inv_sqrt[dst]

    return norm

# Step 5 - gather_source_node_features
def gather_source_node_features(node_features, src):
    # TODO: Return edge-aligned source feature rows (E, F) from node_features.
    return node_features[src]

# Step 6 - scatter_sum_to_nodes
def scatter_sum_to_nodes(edge_features, dst, num_nodes):
    """Scatter-sum edge features onto destination nodes to produce per-node aggregated vectors.

    Args:
        edge_features: FloatTensor of shape (E, F) with one feature row per edge.
        dst: LongTensor of shape (E,) with destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.

    Returns:
        FloatTensor of shape (N, F); row j is the sum of edge features with dst == j.
    """
    # TODO: Scatter-sum edge features onto destination nodes to produce per-node vectors
    out = torch.zeros((num_nodes, edge_features.shape[1]), dtype=edge_features.dtype, device=edge_features.device)
    return out.index_add_(0, dst, edge_features)

# Step 7 - scatter_mean_to_nodes
def scatter_mean_to_nodes(edge_features, dst, num_nodes):
    # Sum edge features onto destination nodes (shape: [N, F])
    summed_features = scatter_sum_to_nodes(edge_features, dst, num_nodes)
    
    # Compute in-degree per destination node (shape: [N])
    degrees = compute_node_degrees(dst, dst, num_nodes)
    
    # Clamp to at least 1.0 to prevent division by zero for isolated nodes,
    # then unsqueeze to [N, 1] for broadcasting across features
    clamped_degrees = degrees.clamp(min=1.0).unsqueeze(-1)
    
    # Divide summed features by node in-degrees
    return summed_features / clamped_degrees

# Step 8 - scatter_max_to_nodes
def scatter_max_to_nodes(edge_features, dst, num_nodes):
    num_edges, num_features = edge_features.shape
    
    # Initialize output tensor filled with -inf using the dtype and device of edge_features
    out = torch.full((num_nodes, num_features), float("-inf"), dtype=edge_features.dtype, device=edge_features.device)
    
    # Early return if there are no edges
    if num_edges == 0:
        return out
        
    # Expand destination indices from (E,) to (E, F) for column-wise scattering
    dst_expanded = dst.unsqueeze(1).expand_as(edge_features)
    
    # Reduce edge features into output tensor using elementwise maximum along dimension 0
    return out.scatter_reduce(0, dst_expanded, edge_features, reduce="amax", include_self=True)

# Step 9 - compute_messages
def compute_messages(node_features, src, dst, message_fn, edge_attr=None):
    """Build per-edge messages via gather + message_fn.

    Args:
        node_features: FloatTensor of shape (N, F).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        message_fn: callable(src_feats, dst_feats[, edge_attr]) -> messages.
        edge_attr: optional FloatTensor of shape (E, Fe).

    Returns:
        messages: FloatTensor of shape (E, M).
    """
    # TODO: Build per-edge messages by gathering features and applying message_fn
    src_feats = gather_source_node_features(node_features, src)
    dst_feats = node_features[dst]
    if edge_attr is None:
        return message_fn(src_feats, dst_feats)
    else:
        return message_fn(src_feats, dst_feats, edge_attr)

# Step 10 - aggregate_messages
def aggregate_messages(messages, dst, num_nodes, aggr='sum'):
    """Aggregate edge messages onto destination nodes using sum, mean, or max.

    Args:
        messages: FloatTensor of shape (E, M) with one message vector per edge.
        dst: LongTensor of shape (E,) with destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.
        aggr: str in {'sum', 'mean', 'max'} selecting the reduction.

    Returns:
        FloatTensor of shape (N, M); row j is the aggregated message for node j.
    """
    # TODO: Aggregate edge messages onto destination nodes via sum/mean/max...
    if aggr == "sum":
        return scatter_sum_to_nodes(messages, dst, num_nodes)
    elif aggr == "mean":
        return scatter_mean_to_nodes(messages, dst, num_nodes)
    elif aggr == "max":
        return scatter_max_to_nodes(messages, dst, num_nodes)
    else:
        raise ValueError

# Step 11 - update_node_features
def update_node_features(node_features, aggregated, update_fn):
    # TODO: Implement update_node_features to fuse each node's current state with its aggregated...
    return update_fn(node_features, aggregated)

# Step 12 - message_passing_layer
def message_passing_layer(node_features, src, dst, message_fn, update_fn, aggr='sum', edge_attr=None):
    """Run one full Gilmer MPNN step: message, aggregate, and update.

    Args:
        node_features: FloatTensor of shape (N, F).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        message_fn: callable(src_feats, dst_feats[, edge_attr]) -> messages (E, M).
        update_fn: callable(node_features, aggregated) -> updated (N, H).
        aggr: str in {'sum', 'mean', 'max'}.
        edge_attr: optional FloatTensor of shape (E, Fe).

    Returns:
        updated_features: FloatTensor of shape (N, H).
    """
    # TODO: compose message, aggregate, and update into one MPNN step
    num_nodes = node_features.shape[0]
    messages = compute_messages(node_features, src, dst, message_fn, edge_attr=edge_attr)
    aggregated = aggregate_messages(messages, dst, num_nodes, aggr=aggr)
    return update_node_features(node_features, aggregated, update_fn)

# Step 13 - stack_message_passing_layers
def stack_message_passing_layers(node_features, src, dst, layers, edge_attr=None):
    """Apply a sequence of message-passing layer callables to produce deep node embeddings.

    Args:
        node_features: FloatTensor of shape (N, F).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        layers: list of callables, each
            layer(node_features, src, dst, edge_attr=None) -> Tensor (N, H_i).
        edge_attr: optional FloatTensor of shape (E, Fe).

    Returns:
        embeddings: FloatTensor of shape (N, H), final layer output.
        all_layer_outputs: list of FloatTensors, one per layer (N, H_i).
    """
    # TODO: Apply a sequence of MP layer callables; return final + intermediates
    h = node_features
    all_layer_outputs = []
    for layer_fn in layers:
        h = layer_fn(h, src, dst, edge_attr=edge_attr)
        all_layer_outputs.append(h)
    return h, all_layer_outputs

# Step 14 - gcn_renormalize_adjacency
def gcn_renormalize_adjacency(src, dst, num_nodes):
    """Apply Kipf-Welling renormalization: self-loops then symmetric norm.

    Args:
        src: LongTensor [E] source node indices.
        dst: LongTensor [E] destination node indices.
        num_nodes: int, number of nodes N.

    Returns:
        src_hat: LongTensor [E + N] sources after self-loops.
        dst_hat: LongTensor [E + N] destinations after self-loops.
        norm_weight: FloatTensor [E + N] symmetrically normalized weights.
    """
    # TODO: add self-loops then symmetrically normalize the adjacency...
    src_hat, dst_hat = add_self_loops(src, dst, num_nodes)
    norm_weight = symmetric_normalize_edge_weights(src_hat, dst_hat, num_nodes, edge_weight=None)
    return src_hat, dst_hat, norm_weight

# Step 15 - gcn_linear_transform
def gcn_linear_transform(node_features, weight, bias=None):
    """Apply the GCN linear feature transform X @ W (+ bias).

    Args:
        node_features: FloatTensor of shape (N, Fin).
        weight: FloatTensor of shape (Fin, Fout).
        bias: optional FloatTensor of shape (Fout).

    Returns:
        FloatTensor of shape (N, Fout).
    """
    # TODO: compute the matrix product and optionally add a bias vector
    if bias is None:
        return node_features @ weight 
    else: 
        return node_features @ weight + bias

# Step 16 - gcn_layer_forward
def gcn_layer_forward(node_features, src, dst, weight, bias=None, num_nodes=None, activation=None):
    """Forward pass of one GCN layer: renormalize, transform, propagate.

    Args:
        node_features: FloatTensor of shape (N, Fin).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        weight: FloatTensor of shape (Fin, Fout).
        bias: optional FloatTensor of shape (Fout,).
        num_nodes: optional int N; defaults to node_features.shape[0].
        activation: optional callable applied to the output.

    Returns:
        FloatTensor of shape (N, Fout).
    """
    # Resolve number of nodes
    N = num_nodes if num_nodes is not None else node_features.shape[0]

    # 1. Renormalize adjacency (adds self-loops and computes symmetric normalization weights)
    src_hat, dst_hat, norm_weight = gcn_renormalize_adjacency(src, dst, N)

    # 2. Linear transformation X * W (without bias yet)
    transformed_features = gcn_linear_transform(node_features, weight, bias=None)

    # 3. Gather source node features for each edge (shape: [E + N, Fout])
    edge_features = gather_source_node_features(transformed_features, src_hat)

    # 4. Scale edge features by symmetric normalization weight (shape: [E + N, Fout])
    message_features = edge_features * norm_weight.unsqueeze(-1)

    # 5. Scatter-sum edge messages onto destination nodes (shape: [N, Fout])
    out = scatter_sum_to_nodes(message_features, dst_hat, N)

    # 6. Add bias after aggregation (if provided)
    if bias is not None:
        out = out + bias

    # 7. Apply optional activation function
    if activation is not None:
        out = activation(out)

    return out

# Step 17 - init_gcn_parameters
def init_gcn_parameters(in_dim, out_dim, with_bias=True, seed=None):
    # TODO: Initialize GCN weight (and optional bias) with Glorot-style uniform...
    if seed is not None:
        torch.manual_seed(seed)
    a = (6/(in_dim + out_dim))**0.5
    weight = torch.empty((in_dim, out_dim)).uniform_(-a, a)
    out = {"weight": weight}
    if with_bias:
        out["bias"] = torch.zeros(out_dim)
    return out

# Step 18 - gcn_stack_forward
def gcn_stack_forward(node_features, src, dst, param_list, activations=None, num_nodes=None):
    """Run a stack of GCN layers to produce deep node embeddings.

    Args:
        node_features: FloatTensor of shape (N, F0).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        param_list: list of dicts, each with 'weight' (Fin, Fout) and optional 'bias' (Fout,).
        activations: optional list of callables or None, one per layer.
        num_nodes: optional int N; defaults to node_features.shape[0].

    Returns:
        embeddings: FloatTensor of shape (N, FL), the final layer output.
        all_layer_outputs: list of FloatTensor outputs after each layer.
    """
    # TODO: Run a stack of GCN layers to produce deep node embeddings
    if num_nodes is None:
        num_nodes = node_features.shape[0]
    if activations is None:
        activations = [None]*len(param_list)
    x = node_features 
    all_layer_outputs = []
    for i, params in enumerate(param_list):
        x = gcn_layer_forward(x, src, dst, weight = params["weight"], bias = params.get("bias", None), num_nodes=None, activation=activations[i])
        all_layer_outputs.append(x)
    return x, all_layer_outputs

# Step 19 - gat_attention_logits
import torch
import torch.nn.functional as F

def gat_attention_logits(node_features, src, dst, attn_src, attn_dst, weight):
    """Compute unnormalized GAT attention logits and transformed features.

    Args:
        node_features: FloatTensor of shape (N, Fin).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        attn_src: FloatTensor of shape (Fout,) source attention vector.
        attn_dst: FloatTensor of shape (Fout,) destination attention vector.
        weight: FloatTensor of shape (Fin, Fout) shared linear transform.

    Returns:
        logits: FloatTensor of shape (E,) unnormalized attention scores.
        transformed: FloatTensor of shape (N, Fout) linearly transformed nodes.
    """
    # 1. Transform node features: H_transformed = X * W (shape: [N, Fout])
    transformed = gcn_linear_transform(node_features, weight, bias=None)

    # 2. Gather source and destination features for each edge (shape: [E, Fout])
    h_src = gather_source_node_features(transformed, src)
    h_dst = gather_source_node_features(transformed, dst)

    # 3. Compute dot products across feature dimension Fout (shape: [E])
    score_src = (h_src * attn_src).sum(dim=-1)
    score_dst = (h_dst * attn_dst).sum(dim=-1)

    # 4. Sum source and destination scores and apply LeakyReLU(0.2)
    unnormalized_scores = score_src + score_dst
    logits = F.leaky_relu(unnormalized_scores, negative_slope=0.2)

    return logits, transformed

# Step 20 - gat_masked_neighbor_softmax
import torch

def gat_masked_neighbor_softmax(logits, dst, num_nodes):
    """Numerically stable softmax of attention logits over each dest node's neighbors.

    Args:
        logits: FloatTensor of shape (E,) with one unnormalized attention logit per edge.
        dst: LongTensor of shape (E,) with destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.

    Returns:
        FloatTensor of shape (E,) with attention coefficients that sum to 1 over
        each destination's incoming edges.
    """
    if logits.numel() == 0:
        return logits

    # 1. Reshape logits to (E, 1) for feature-compatible scatter helpers
    logits_2d = logits.unsqueeze(-1)

    # 2. Compute max logit per destination node for numerical stability (shape: [N, 1])
    node_max = scatter_max_to_nodes(logits_2d, dst, num_nodes)

    # 3. Gather per-node max to each edge and compute stabilized exp(logits - max)
    edge_max = node_max[dst]
    exp_logits = torch.exp(logits_2d - edge_max)

    # 4. Sum stabilized exp values per destination node (normalizer Z_j, shape: [N, 1])
    node_exp_sum = scatter_sum_to_nodes(exp_logits, dst, num_nodes)

    # 5. Gather normalizers back to edges and divide
    edge_exp_sum = node_exp_sum[dst]
    alpha_2d = exp_logits / edge_exp_sum

    # 6. Squeeze back to shape (E,)
    return alpha_2d.squeeze(-1)

# Step 21 - gat_head_forward
def gat_head_forward(node_features, src, dst, weight, attn_src, attn_dst, bias=None, num_nodes=None, activation=None):
    """Forward pass of a single GAT attention head.

    Args:
        node_features: FloatTensor of shape (N, Fin).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        weight: FloatTensor of shape (Fin, Fout) shared linear transform.
        attn_src: FloatTensor of shape (Fout,) source attention vector.
        attn_dst: FloatTensor of shape (Fout,) destination attention vector.
        bias: optional FloatTensor of shape (Fout,).
        num_nodes: optional int N; inferred from node_features if None.
        activation: optional callable applied to the head output.

    Returns:
        head_out: FloatTensor of shape (N, Fout).
        attn_coeffs: FloatTensor of shape (E,) attention coefficients.
    """
    # 1. Resolve number of nodes
    if num_nodes is None:
        num_nodes = node_features.shape[0]

    # 2. Compute unnormalized attention logits and transformed node features XW
    logits, transformed = gat_attention_logits(node_features, src, dst, attn_src, attn_dst, weight)

    # 3. Softmax normalize edge logits per destination node (shape: [E])
    attn_coeffs = gat_masked_neighbor_softmax(logits, dst, num_nodes)

    # 4. Gather source node features for each edge (shape: [E, Fout])
    h_src = gather_source_node_features(transformed, src)

    # 5. Weight source features by attention coefficients (shape: [E, Fout])
    weighted_messages = h_src * attn_coeffs.unsqueeze(-1)

    # 6. Scatter-sum weighted messages onto destination nodes (shape: [N, Fout])
    head_out = scatter_sum_to_nodes(weighted_messages, dst, num_nodes)

    # 7. Add optional bias
    if bias is not None:
        head_out = head_out + bias

    # 8. Apply optional activation function
    if activation is not None:
        head_out = activation(head_out)

    return head_out, attn_coeffs

# Step 22 - merge_gat_heads
import torch

def merge_gat_heads(head_outputs, mode='concat'):
    """Merge multi-head GAT outputs into one node-feature tensor.

    Args:
        head_outputs: A list/tuple of FloatTensors, each of shape (N, F),
                      OR a single FloatTensor of shape (H, N, F).
        mode: Merging strategy - 'concat' or 'mean'.

    Returns:
        FloatTensor of shape (N, H * F) if mode='concat',
        or shape (N, F) if mode='mean'.
    """
    if mode not in ('concat', 'mean'):
        raise ValueError(f"Unsupported mode '{mode}'. Expected 'concat' or 'mean'.")

    # If list/tuple of (N, F) tensors, merge directly or convert to (H, N, F)
    if isinstance(head_outputs, (list, tuple)):
        if mode == 'concat':
            return torch.cat(head_outputs, dim=-1)
        elif mode == 'mean':
            stacked = torch.stack(head_outputs, dim=0)
            return stacked.mean(dim=0)

    # If head_outputs is already a 3D Tensor of shape (H, N, F)
    elif isinstance(head_outputs, torch.Tensor):
        if mode == 'concat':
            H, N, F = head_outputs.shape
            # Permute to (N, H, F) then reshape to (N, H * F)
            return head_outputs.permute(1, 0, 2).reshape(N, H * F)
        elif mode == 'mean':
            return head_outputs.mean(dim=0)

    else:
        raise TypeError("head_outputs must be a list/tuple of tensors or a 3D Tensor.")

# Step 23 - gat_layer_forward
def gat_layer_forward(node_features, src, dst, head_params, merge_mode='concat', num_nodes=None, activation=None):
    """Multi-head GAT layer: run each head, merge, optional activation.

    Args:
        node_features: FloatTensor (N, Fin).
        src: LongTensor (E,) source indices.
        dst: LongTensor (E,) destination indices.
        head_params: list of dicts with keys weight, attn_src, attn_dst,
            and optional bias for each head.
        merge_mode: 'concat' or 'mean'.
        num_nodes: optional int N; inferred from node_features if None.
        activation: optional callable applied after merging heads.

    Returns:
        out: FloatTensor (N, F_merged).
        all_attn: list of FloatTensor (E,) attention coeffs per head.
    """
    if num_nodes is None:
        num_nodes = node_features.shape[0]

    head_outputs = []
    all_attn = []

    # 1. Run each attention head without applying per-head activations
    for params in head_params:
        head_out, attn_coeffs = gat_head_forward(
            node_features,
            src,
            dst,
            weight=params["weight"],
            attn_src=params["attn_src"],
            attn_dst=params["attn_dst"],
            bias=params.get("bias", None),
            num_nodes=num_nodes,
            activation=None,  # Activation is applied after merging across heads
        )
        head_outputs.append(head_out)
        all_attn.append(attn_coeffs)

    # 2. Merge attention heads via 'concat' or 'mean'
    out = merge_gat_heads(head_outputs, mode=merge_mode)

    # 3. Apply post-merge layer activation
    if activation is not None:
        out = activation(out)

    return out, all_attn

# Step 24 - init_gat_parameters
import math
import torch


def init_gat_parameters(
    in_dim, out_dim, num_heads=1, with_bias=True, seed=None
):
    if seed is not None:
        torch.manual_seed(seed)

    # Calculate Glorot uniform limits: a = sqrt(6 / (fan_in + fan_out))
    limit_w = math.sqrt(6.0 / (in_dim + out_dim))
    limit_a = math.sqrt(6.0 / (out_dim + 1))

    heads = []
    for _ in range(num_heads):
        # Sample uniformly from [-limit, limit) via (rand * 2 - 1) * limit
        w = (
            (torch.rand(in_dim, out_dim) * 2 - 1) * limit_w
        ).requires_grad_(True)
        attn_src = (
            (torch.rand(out_dim) * 2 - 1) * limit_a
        ).requires_grad_(True)
        attn_dst = (
            (torch.rand(out_dim) * 2 - 1) * limit_a
        ).requires_grad_(True)

        head_dict = {
            'weight': w,
            'attn_src': attn_src,
            'attn_dst': attn_dst,
        }

        if with_bias:
            head_dict['bias'] = torch.zeros(out_dim, requires_grad=True)

        heads.append(head_dict)

    return heads

# Step 25 - gat_stack_forward
def gat_stack_forward(node_features, src, dst, layer_param_list, merge_modes=None, activations=None, num_nodes=None):
    """Run a stack of multi-head GAT layers.

    Args:
        node_features: FloatTensor (N, F0).
        src: LongTensor (E,) source indices.
        dst: LongTensor (E,) destination indices.
        layer_param_list: list of length L; each entry is a head_params list
            for gat_layer_forward.
        merge_modes: optional list of L merge mode strings ('concat' or 'mean').
            Defaults to 'concat' for every layer.
        activations: optional list of L callables or None. Defaults to no
            activation for every layer.
        num_nodes: optional int N; inferred from node_features if None.

    Returns:
        embeddings: FloatTensor (N, FL) final layer output.
        all_layer_outputs: list of L FloatTensors, the output after each layer.
    """
    # TODO: Run a stack of multi-head GAT layers for deep node embeddings.
    if num_nodes is None:
        num_nodes = node_features.shape[0]
    L = len(layer_param_list)
    if merge_modes is None:
        merge_modes = ["concat"]*L 
    if activations is None:
        activations = [None]*L 
    x = node_features 
    all_layer_outputs = []
    for i in range(L):
        x, _ = gat_layer_forward(x, src, dst, layer_param_list[i], merge_modes[i], num_nodes, activations[i])
        all_layer_outputs.append(x)
    return x, all_layer_outputs

# Step 26 - global_mean_pool
def global_mean_pool(node_features, batch_index, num_graphs=None):
    """Globally mean-pool node features into one graph-level vector per graph.

    Args:
        node_features: FloatTensor of shape (N, F) with one feature row per node.
        batch_index: LongTensor of shape (N,) mapping each node to a graph id in
            {0, ..., B-1}.
        num_graphs: Optional int B. If None, inferred as batch_index.max() + 1.

    Returns:
        FloatTensor of shape (B, F); row b is the mean of node features with
        batch_index == b.
    """
    # TODO: Mean-pool node features into one graph-level vector per graph...
    if num_graphs is None:
        B = int(batch_index.max())+1
    else:
        B = num_graphs
    scatter = scatter_sum_to_nodes(node_features, batch_index, B)
    count = torch.ones(len(node_features), 1, dtype=node_features.dtype, device=node_features.device)
    count_sc = scatter_sum_to_nodes(count, batch_index, B)
    return scatter/count_sc

# Step 27 - global_sum_pool
def global_sum_pool(node_features, batch_index, num_graphs=None):
    """Globally sum-pool node features into one graph-level vector per graph.

    Args:
        node_features: FloatTensor of shape (N, F) with one row per node.
        batch_index: LongTensor of shape (N,) mapping each node to a graph id
            in 0 .. B-1.
        num_graphs: optional int B. If None, inferred as max(batch_index) + 1.

    Returns:
        FloatTensor of shape (B, F); row g is the sum of node features with
        batch_index == g.
    """
    # TODO: sum-pool node features into one graph-level vector per graph
    if num_graphs is None:
        num_graphs = int(batch_index.max().item())+1 
    return scatter_sum_to_nodes(node_features, batch_index, num_graphs)

# Step 28 - global_max_pool
def global_max_pool(node_features, batch_index, num_graphs=None):
    # TODO: Globally max-pool node features into one graph-level vector per graph.
    if num_graphs is None:
        num_graphs = int(batch_index.max().item())+1
    return scatter_max_to_nodes(node_features, batch_index, num_graphs)

# Step 29 - global_mean_max_pool
def global_mean_max_pool(node_features, batch_index, num_graphs=None):
    """Concatenate global mean and max pooled features into a 2F-dim graph vector.

    Args:
        node_features: FloatTensor of shape (N, F).
        batch_index: LongTensor of shape (N,) with graph ids in {0, ..., B-1}.
        num_graphs: Optional int B. If None, inferred as batch_index.max() + 1.

    Returns:
        FloatTensor of shape (B, 2F); each row is [mean_pool || max_pool].
    """
    # TODO: Concatenate global mean and max pooled features into a 2F-dim vector...
    mean = global_mean_pool(node_features, batch_index, num_graphs)
    maxi = global_max_pool(node_features, batch_index, num_graphs)
    return torch.cat([mean, maxi], dim=-1)

# Step 30 - node_classification_head
def node_classification_head(node_embeddings, weight, bias=None):
    # TODO: Map node embeddings to per-node class logits via a linear head...
    out = node_embeddings @ weight 
    if bias is not None:
        out += bias 
    return out

# Step 31 - graph_regression_head (not yet solved)
# TODO: implement

# Step 32 - generate_sbm_graph
import torch


def generate_sbm_graph(
    num_nodes, num_classes, p_in, p_out, feature_dim, seed=None
):
    if seed is not None:
        torch.manual_seed(seed)

    # 1. Assign community labels in contiguous blocks
    node_labels = torch.zeros(num_nodes, dtype=torch.long)
    for c in range(num_classes):
        start_idx = c * num_nodes // num_classes
        end_idx = (c + 1) * num_nodes // num_classes
        node_labels[start_idx:end_idx] = c

    # 2. Sample random node features
    node_features = torch.randn(num_nodes, feature_dim)

    # 3. Sample edges for each unordered pair (i < j)
    src_list = []
    dst_list = []

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            p = p_in if node_labels[i] == node_labels[j] else p_out
            if torch.rand(1).item() < p:
                # Add both directions for undirected representation
                src_list.extend([i, j])
                dst_list.extend([j, i])

    # 4. Construct edge_index in COO format (2, E)
    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "node_labels": node_labels,
        "num_nodes": num_nodes,
    }

# Step 33 - build_node_classification_dataset
def build_node_classification_dataset(num_graphs, num_nodes, num_classes, p_in, p_out, feature_dim, seed=None):
    # TODO: Build a list of SBM graphs with consistent schema for node classification.
    graphs = []
    for i in range(num_graphs):
        if seed is not None:
            tempseed = seed + i 
        else:
            tempseed = None 
        graph = generate_sbm_graph(num_nodes, num_classes, p_in, p_out, feature_dim, seed=tempseed)
        graphs.append(graph)
    return graphs

# Step 34 - generate_molecule_like_graph
import torch


def generate_molecule_like_graph(
    num_nodes, num_node_features, edge_prob=0.3, seed=0
):
    if seed is not None:
        torch.manual_seed(seed)

    # 1. Sample continuous node features (N, F)
    x = torch.randn(num_nodes, num_node_features)

    # 2. Sample undirected edges without self-loops using an upper-triangular mask
    rand_matrix = torch.rand(num_nodes, num_nodes)
    upper_tri_mask = torch.triu(rand_matrix < edge_prob, diagonal=1)
    src_upper, dst_upper = torch.nonzero(upper_tri_mask, as_tuple=True)

    if src_upper.numel() > 0:
        # Concatenate both directions for undirected representation
        src = torch.cat([src_upper, dst_upper], dim=0)
        dst = torch.cat([dst_upper, src_upper], dim=0)
        edge_index = torch.stack([src, dst], dim=0).long()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # 3. Compute node degrees
    if edge_index.numel() > 0:
        deg = torch.bincount(edge_index[0], minlength=num_nodes).float()
    else:
        deg = torch.zeros(num_nodes, dtype=torch.float)

    # 4. Compute graph-level regression target: y = (1/N) * sum_v (deg(v) * x_bar_v)
    x_bar = x.mean(dim=1)  # (N,)
    y = (deg * x_bar).mean()  # 0-dim scalar tensor

    return {
        "x": x,
        "edge_index": edge_index,
        "y": y,
    }

# Step 35 - build_graph_regression_dataset
def build_graph_regression_dataset(num_graphs, num_nodes_range, num_node_features, edge_prob=0.3, seed=0):
    # TODO: Build a list of molecule-like graphs for graph-level regression.
    graphs = []
    lo, hi = num_nodes_range
    for i in range(num_graphs):
        num_nodes = lo + (i % (hi - lo + 1))
        graph = generate_molecule_like_graph(num_nodes, num_node_features, edge_prob=edge_prob, seed=seed+i)
        graphs.append(graph)
    return graphs

# Step 36 - collate_graph_batch
import torch


def collate_graph_batch(graphs):
    x_list = []
    edge_index_list = []
    batch_list = []
    y_list = []

    offset = 0

    for i, g in enumerate(graphs):
        x = g["x"]
        n = x.shape[0]

        # 1. Accumulate node features
        x_list.append(x)

        # 2. Shift edge indices by current node offset and accumulate
        edge_index_shifted = g["edge_index"] + offset
        edge_index_list.append(edge_index_shifted)

        # 3. Create batch indicator vector for current graph nodes
        batch_list.append(torch.full((n,), i, dtype=torch.long))

        # 4. Ensure scalar 0-dim float tensor for target y
        y = g["y"]
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
        else:
            y = y.float().reshape(())
        y_list.append(y)

        offset += n

    # Concatenate / stack accumulated tensors
    x_batch = torch.cat(x_list, dim=0)
    edge_index_batch = (
        torch.cat(edge_index_list, dim=1)
        if edge_index_list
        else torch.empty((2, 0), dtype=torch.long)
    )
    batch_vec = torch.cat(batch_list, dim=0)
    y_batch = torch.stack(y_list, dim=0)

    return {
        "x": x_batch,
        "edge_index": edge_index_batch,
        "batch": batch_vec,
        "y": y_batch,
    }

# Step 37 - cross_entropy_loss
import torch

def cross_entropy_loss(logits, targets):
    # Log-softmax over the class dimension
    log_probs = torch.log_softmax(logits, dim=-1)

    # Gather the log-probabilities corresponding to the ground-truth targets
    true = log_probs[torch.arange(logits.shape[0]), targets]
    
    # Return the negative mean across the batch
    return -true.mean()

# Step 38 - mse_loss
def mse_loss(predictions, targets):
    # TODO: Compute mean squared error between predictions and targets
    pred, tar = predictions.view(-1), targets.view(-1)
    return ((pred-tar)**2).mean()

# Step 39 - accuracy_metric
def accuracy_metric(logits, targets):
    # TODO: Return the fraction of argmax(logits) predictions matching targets.
    guess = torch.argmax(logits, dim=-1)
    return (targets == guess).float().mean()

# Step 40 - mae_metric
def mae_metric(predictions, targets):
    # TODO: Compute mean absolute error between predicted and target continuous values.
    pred, tar = predictions.view(-1), targets.view(-1)
    return torch.abs(pred-tar).mean().item()

# Step 41 - gnn_train_step
import torch

def gnn_train_step(params, batch, forward_fn, loss_fn, lr):
    # 1. Forward pass
    predictions = forward_fn(params, batch)
    targets = batch["y"]
    loss = loss_fn(predictions, targets)

    # 2. Backward pass
    loss.backward()

    # 3. In-place SGD parameter update & gradient reset
    with torch.no_grad():
        for tensor in params.values():
            if tensor.grad is not None:
                tensor.sub_(lr * tensor.grad)  # In-place SGD update
                tensor.grad.zero_()             # Reset gradient for next step

    return {"loss": loss.item(), "params": params}

# Step 42 - train_node_classifier
import torch

def train_node_classifier(params, dataset, forward_fn, num_epochs, lr, mask_key='train_mask'):
    x = dataset["x"]
    edge_index = dataset["edge_index"]
    mask = dataset[mask_key]
    y_masked = dataset["y"][mask]

    # Prepare batch with masked targets
    batch = {
        "x": x,
        "edge_index": edge_index,
        "y": y_masked,
        "mask": mask
    }

    # Wrap forward_fn to accept (params, batch) and return masked logits
    def wrapped_forward(p, b):
        full_logits = forward_fn(p, b["x"], b["edge_index"])
        return full_logits[b["mask"]]

    history = []

    for _ in range(num_epochs):
        # 1. Training step
        out = gnn_train_step(params, batch, wrapped_forward, cross_entropy_loss, lr)
        params = out["params"]
        step_loss = float(out["loss"])

        # 2. Evaluation step under no_grad
        with torch.no_grad():
            eval_logits = forward_fn(params, x, edge_index)[mask]
            acc = float(accuracy_metric(eval_logits, y_masked))

        history.append({"loss": step_loss, "accuracy": acc})

    return {"history": history, "params": params}

# Step 43 - train_graph_regressor
import torch


def train_graph_regressor(
    params, graphs, forward_fn, num_epochs, lr, batch_size=8
):
    """Train a graph regressor over multiple epochs of mini-batches.

    Args:
        params: dict of trainable torch tensors.
        graphs: list of graph dicts with keys x, edge_index, y.
        forward_fn: callable(params, batch) -> predictions.
        num_epochs: number of training epochs.
        lr: learning rate for SGD updates.
        batch_size: graphs per mini-batch (default 8).

    Returns:
        history: dict with 'loss' and 'mae' lists of per-epoch floats.
        params: updated parameter dict.
    """
    history = {"loss": [], "mae": []}
    num_graphs = len(graphs)

    # Pre-collate full dataset for whole-dataset MAE evaluation
    full_batch = collate_graph_batch(graphs)

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        # Shuffle indices per epoch
        perm = torch.randperm(num_graphs).tolist()
        shuffled_graphs = [graphs[i] for i in perm]

        # Mini-batch training loop
        for start_idx in range(0, num_graphs, batch_size):
            batch_graphs = shuffled_graphs[start_idx : start_idx + batch_size]
            batch = collate_graph_batch(batch_graphs)

            step_result = gnn_train_step(
                params, batch, forward_fn, mse_loss, lr
            )
            params = step_result["params"]
            total_loss += float(step_result["loss"])
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Full-dataset MAE evaluation
        with torch.no_grad():
            preds = forward_fn(params, full_batch)
            epoch_mae = float(mae_metric(preds, full_batch["y"]))

        history["loss"].append(avg_loss)
        history["mae"].append(epoch_mae)

    return history, params

# Step 44 - representation_similarity (not yet solved)
# TODO: implement

# Step 45 - oversmoothing_diagnostic (not yet solved)
# TODO: implement

# Step 46 - mpnn_gnn_experiment (not yet solved)
# TODO: implement

