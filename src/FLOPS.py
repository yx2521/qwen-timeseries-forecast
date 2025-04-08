# FLOPS Calculation for Qwen 2.5

# 1. Map the Token Embeddings
#（embed_tokens): Embedding(151936, 896)
# No FLOPS for this step

# 2. Positional Encoding
# Include the cost of adding the positional encodings to the token embeddings
def positional_encoding_flops(in_dim):
    """
    Calculate the FLOPs required for adding positional encodings to token embeddings.

    Args:
        in_dim (tuple): A tuple (D, N) where:
            D (int): Dimension of each token embedding.
            N (int): Sequence length (number of tokens).

    Returns:
        int: Total FLOPs for positional encoding.
    """
    D, N = in_dim[0], in_dim[1]
    flops = D * N
    return flops


# 3. 24 Layers of self-attention, MLP, and normalization

def matrix_mul_without_bias(matrix_1_dim, matrix_2_dim):
    """
    Calculate the FLOPs for matrix multiplication without bias.

    Args:
        matrix_1_dim (tuple): Dimensions of the first matrix (m1_m, m1_n).
        matrix_2_dim (tuple): Dimensions of the second matrix (m2_n, m2_p).

    Returns:
        int: Total FLOPs for the matrix multiplication.
    """
    m1_m, m1_n = matrix_1_dim[0], matrix_1_dim[1]
    m2_n, m2_p = matrix_2_dim[0], matrix_2_dim[1]
    flops = m1_m * m2_p * (2 * m1_n - 1)
    return flops

def matrix_mul_with_bias(matrix_1_dim, matrix_2_dim):
    """
    Calculate the FLOPs for matrix multiplication with bias.

    Args:
        matrix_1_dim (tuple): Dimensions of the first matrix (m1_m, m1_n).
        matrix_2_dim (tuple): Dimensions of the second matrix (m2_n, m2_p).

    Returns:
        int: Total FLOPs for the matrix multiplication followed by addition of bias.
    """
    m1_m, m1_n = matrix_1_dim[0], matrix_1_dim[1]
    m2_n, m2_p = matrix_2_dim[0], matrix_2_dim[1]
    flops = m1_m * m2_p * (2 * m1_n - 1) + m1_m * m2_p
    return flops


    ## Input Layernorm: RMSNorm

def RMSNorm_flops(in_dim):
    """
    Calculate the FLOPs required for RMSNorm (Root Mean Square Normalization).

    Args:
        in_dim (tuple): A tuple (D, N) where:
            D (int): Dimension of each token embedding.
            N (int): Sequence length (number of tokens).

    Returns:
        int: Total FLOPs for RMSNorm.
    """
    D, N = in_dim[0], in_dim[1]
    flops_element_squares = D * N  # x^2 for all elements
    flops_column_sqr_sum = N * (D - 1)  # for each token, sum x^2 over D
    flops_root = N * 2 + N * 10  # for each column sum, divide by D, add ε, then take square root
    flops_divide = D * N   # element-wise x / norm

    flops = flops_element_squares + flops_column_sqr_sum + flops_root + flops_divide

    return flops


    ## Self-Attention
    # Assume the FLOPS for this operation are the same as a standard multi-head attention

def attention_flops(in_dim, num_heads):
    """
    Estimate FLOPs for a standard multi-head attention block with residual connection.
    
    Args:
        in_dim: Tuple[int, int] = (D, N) where
            D = dimension of each token embedding,
            N = sequence length (number of tokens)
        num_heads: int = number of attention heads

    Returns:
        Total FLOPs for one self-attention block
    """
    D, N, H = in_dim[0], in_dim[1], num_heads
    Dh = D // H  # per-head dimension

    # Input shape for each head: [Dh, N]
    in_dim_mh = [Dh, N]
    weight_dim = [Dh, Dh]

    # Linear projections for Q, K, V: [D/H, D/H] × [D/H, N] + bias
    flops_q_proj = matrix_mul_with_bias(weight_dim, in_dim_mh)
    flops_k_proj = matrix_mul_with_bias(weight_dim, in_dim_mh)
    flops_v_proj = matrix_mul_with_bias(weight_dim, in_dim_mh)

    # Kᵀ × Q: [N, D/H] × [D/H, N] = [N, N]
    flops_qk = matrix_mul_without_bias([N, Dh], [Dh, N])

    # Softmax over [N, N]
    flops_softmax = 10 * (N**2) + N * (N - 1) + N**2

    # Attention-weighted V: [Dh, N] × [N, N] = [Dh, N]
    flops_att_v = matrix_mul_without_bias(in_dim_mh, [N, N])

    # Output projection after concatenating all heads: [D, D] × [D, N]
    flops_output_proj = matrix_mul_without_bias([D, D], [D, N])

    # Residual connection: x + output, element-wise add
    flops_residual = D * N

    # Total FLOPs
    flops_total = H * (flops_q_proj + flops_k_proj + flops_v_proj + flops_qk + flops_softmax + flops_att_v)
    flops_total += flops_output_proj + flops_residual

    return flops_total


    ## Post-attention Layernorm: RMSNorm
    # Apply the RMSNorm calculation defined above


    ## MLP: SwiGLU activation

def mlp_flops(in_dim, hidden_dim):
    """
    Calculate the FLOPs required for the MLP (Multi-Layer Perceptron) with SwiGLU activation.

    Args:
        in_dim (tuple): A tuple (D, N) where:
            D (int): Dimension of each token embedding.
            N (int): Sequence length (number of tokens).
        hidden_dim (int): Dimension of the hidden layer in the MLP.

    Returns:
        int: Total FLOPs for the MLP with SwiGLU activation.
    """
    D, N = in_dim[0], in_dim[1]

    flops_up_proj = (2 * D - 1) * hidden_dim
    flops_gate_proj = (2 * D - 1) * hidden_dim

    # SiLU = x * sigmoid(x) --> 1 multiplication, 1 exponential, 1 addition, 1 inverse, 1 multiplication
    flops_SiLU = (1+10+1+1+1) * hidden_dim

    flops_up_times_gate = hidden_dim

    flops_down_proj = (2 * hidden_dim - 1) * D

    flops_per_token = flops_up_proj + flops_gate_proj + flops_SiLU + flops_up_times_gate + flops_down_proj

    # Residual connection
    flops_res = D * N

    flop_sum = flops_per_token * N + flops_res 

    return flop_sum


    ## Sum Attention and MLP FLOPs for one system

def transformer_block_flops(in_dim, num_heads, hidden_dim, num_layers=24):
    """
    Calculate the total FLOPs for a transformer block, including attention, MLP, and layer normalization.

    Args:
        in_dim (tuple): A tuple (D, N) where:
            D (int): Dimension of each token embedding.
            N (int): Sequence length (number of tokens).
        num_heads (int): Number of attention heads.
        hidden_dim (int): Dimension of the hidden layer in the MLP.
        num_layers (int, optional): Number of transformer layers. Defaults to 24.

    Returns:
        int: Total FLOPs for the transformer block across all layers.
    """
    flops = attention_flops(in_dim, num_heads) + mlp_flops(in_dim, hidden_dim) + RMSNorm_flops(in_dim)*2
    return flops * num_layers


# 4. Final Normalisation and LM Head

    ## Final LayerNorm: RMSNorm
    # Apply the RMSNorm calculation defined above


    ## LM Head: Linear projection
def LM_head_flops(in_dim, out_features):
    """
    Calculate the FLOPs for the LM (Language Model) head, which includes a linear projection.

    Args:
        in_dim (tuple): A tuple (D, N) where:
            D (int): Dimension of each token embedding.
            N (int): Sequence length (number of tokens).
        out_features (int): Number of output features (vocabulary size).

    Returns:
        int: Total FLOPs for the LM head.
    """
    # Linear projection: [V, D] × [D, N] + bias
    weight_dim = [out_features , in_dim[0]]
    flops = matrix_mul_with_bias(weight_dim, in_dim)
    return flops



# 5. Total FLOPs for forward pass

def flops_qwen_forward(batch_size, in_dim, num_heads, hidden_dim, out_features):
    """
    Calculate the total FLOPs for both forward and backward passes of the Qwen model.

    Args:
        batch_size (int): Number of samples in a batch.
        in_dim (tuple): A tuple (D, N) where:
            D (int): Dimension of each token embedding.
            N (int): Sequence length (number of tokens).
        num_heads (int): Number of attention heads.
        hidden_dim (int): Dimension of the hidden layer in the MLP.
        out_features (int): Number of output features (vocabulary size).

    Returns:
        int: Total FLOPs for forward and backward passes.
    """
    flops_sum = positional_encoding_flops(in_dim) + transformer_block_flops(in_dim, num_heads, hidden_dim) + RMSNorm_flops(in_dim) + LM_head_flops(in_dim, out_features)

    return flops_sum * batch_size


# 6. Total FLOPs for the experiments
def flop_forward_and_backward(batch_size, in_dim, num_heads, hidden_dim, out_features):
    """
    Calculate the total FLOPs for both forward and backward passes of the Qwen model.

    Args:
        batch_size (int): Number of samples in a batch.
        in_dim (tuple): A tuple (D, N) where:
            D (int): Dimension of each token embedding.
            N (int): Sequence length (number of tokens).
        num_heads (int): Number of attention heads.
        hidden_dim (int): Dimension of the hidden layer in the MLP.
        out_features (int): Number of output features (vocabulary size).

    Returns:
        int: Total FLOPs for forward and backward passes.
    """
    return 3 * flops_qwen_forward(batch_size, in_dim, num_heads, hidden_dim, out_features)

def total_flops(no_steps, batch_size, in_dim, num_heads, hidden_dim, out_features):
    """
    Calculate the total FLOPs for the entire training process.

    Args:
        no_steps (int): Number of training steps.
        batch_size (int): Number of samples in a batch.
        in_dim (tuple): A tuple (D, N) where:
            D (int): Dimension of each token embedding.
            N (int): Sequence length (number of tokens).
        num_heads (int): Number of attention heads.
        hidden_dim (int): Dimension of the hidden layer in the MLP.
        out_features (int): Number of output features (vocabulary size).

    Returns:
        int: Total FLOPs for the training process.
    """
    return no_steps * flop_forward_and_backward(batch_size, in_dim, num_heads, hidden_dim, out_features)


# 7. Add the FLOPs for LoRA
def lora_flops_forward(in_dim, num_heads, r, num_layers = 24):
    """
    Calculate the FLOPs for the LoRA (Low-Rank Adaptation) forward pass.

    Args:
        in_dim (tuple): A tuple (D, N) where:
            D (int): Dimension of each token embedding.
            N (int): Sequence length (number of tokens).
        num_heads (int): Number of attention heads.
        r (int): Rank of the low-rank adaptation matrices.
        num_layers (int, optional): Number of transformer layers. Defaults to 24.

    Returns:
        int: Total FLOPs for the LoRA forward pass across all layers.
    """    
    D, N, H = in_dim[0], in_dim[1], num_heads
    Dh = D // H  # per-head dimension

    # Input shape for each head: [Dh, N]
    in_dim_mh = [Dh, N]
    A_dim = [r, Dh]
    Ax_dim = [r, N]
    B_dim = [Dh, r]
    
    flops_lora_A = matrix_mul_without_bias(A_dim, in_dim_mh)
    flops_lora_B = matrix_mul_without_bias(B_dim, Ax_dim)

    # Additional scaling with alpha/r and add back
    flops_lora_output = Dh * N * 3

    # Multiply by 2 because LoRA is applied to both q_proj and v_proj
    return H * (flops_lora_A + flops_lora_B + flops_lora_output) * num_layers * 2

# 8. Total FLOPs with LoRA
def total_flops_lora(no_steps, batch_size, in_dim, num_heads, hidden_dim, out_features, r):
    """
    Calculate the total FLOPs for the entire training process with LoRA.

    Args:
        no_steps (int): Number of training steps.
        batch_size (int): Number of samples in a batch.
        in_dim (tuple): A tuple (D, N) where:
            D (int): Dimension of each token embedding.
            N (int): Sequence length (number of tokens).
        num_heads (int): Number of attention heads.
        hidden_dim (int): Dimension of the hidden layer in the MLP.
        out_features (int): Number of output features (vocabulary size).
        r (int): Rank of the low-rank adaptation matrices.

    Returns:
        int: Total FLOPs for the training process with LoRA.
    """
    return total_flops(no_steps, batch_size, in_dim, num_heads, hidden_dim, out_features) + lora_flops_forward(in_dim, num_heads, r) * batch_size * no_steps * 3

