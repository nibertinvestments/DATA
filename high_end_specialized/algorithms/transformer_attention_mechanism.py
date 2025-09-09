"""
Transformer Attention Mechanism Algorithm
========================================

Complete implementation of the multi-head self-attention mechanism from the
"Attention Is All You Need" paper, including positional encoding, scaled
dot-product attention, and feed-forward networks.

Mathematical Foundation:
Attention(Q,K,V) = softmax(QK^T/√d_k)V

Multi-Head Attention:
MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

Applications:
- Natural Language Processing
- Machine Translation
- Text Summarization
- Computer Vision
- Time Series Analysis
- Reinforcement Learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Configuration for Transformer model."""
    d_model: int = 512          # Model dimension
    n_heads: int = 8            # Number of attention heads
    d_ff: int = 2048            # Feed-forward dimension
    max_seq_length: int = 5000  # Maximum sequence length
    dropout: float = 0.1        # Dropout probability
    layer_norm_eps: float = 1e-6
    vocab_size: int = 10000     # Vocabulary size


class PositionalEncoding(nn.Module):
    """
    Positional encoding using sinusoidal functions.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        
        # Calculate div_term for even and odd indices
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        # Apply sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        seq_length = x.size(1)
        return x + self.pe[:, :seq_length]


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    
    Attention(Q,K,V) = softmax(QK^T/√d_k)V
    """
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, seq_len, d_k]
            key: [batch_size, seq_len, d_k]
            value: [batch_size, seq_len, d_v]
            mask: [batch_size, seq_len, seq_len] or broadcastable
            
        Returns:
            output: [batch_size, seq_len, d_v]
            attention_weights: [batch_size, seq_len, seq_len]
        """
        d_k = query.size(-1)
        
        # Calculate attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len]
            
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear projections and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Expand mask for multi-head attention
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        
        # Apply attention
        attn_output, attention_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        output = self.w_o(attn_output)
        output = self.dropout(output)
        
        return output, attention_weights


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer encoder layer with self-attention and feed-forward.
    
    Uses pre-norm architecture: LayerNorm -> MultiHeadAttention -> Residual
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attention = MultiHeadAttention(
            config.d_model, config.n_heads, config.dropout
        )
        self.feed_forward = PositionwiseFeedForward(
            config.d_model, config.d_ff, config.dropout
        )
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len]
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # Pre-norm self-attention with residual connection
        norm_x = self.norm1(x)
        attn_output, _ = self.self_attention(norm_x, norm_x, norm_x, mask)
        x = x + self.dropout(attn_output)
        
        # Pre-norm feed-forward with residual connection
        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout(ff_output)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Complete Transformer encoder with multiple layers.
    """
    
    def __init__(self, config: TransformerConfig, num_layers: int = 6):
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(config.d_model, config.max_seq_length)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config) for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_padding_mask(self, x: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
        """Create padding mask to ignore padded tokens."""
        # Create mask where True indicates real tokens, False indicates padding
        mask = (x != pad_token_id).unsqueeze(1).unsqueeze(2)
        return mask.float()
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (1 for real tokens, 0 for padding)
            
        Returns:
            Dictionary with 'last_hidden_state' and 'attention_weights'
        """
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = self.create_padding_mask(input_ids)
        else:
            # Expand attention mask to [batch_size, 1, 1, seq_len] for broadcasting
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Token embeddings with positional encoding
        x = self.token_embedding(input_ids) * math.sqrt(self.config.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Store attention weights from all layers
        all_attention_weights = []
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # Final layer normalization
        x = self.final_norm(x)
        
        return {
            'last_hidden_state': x,
            'attention_weights': all_attention_weights
        }


class TransformerClassifier(nn.Module):
    """
    Transformer-based sequence classifier.
    """
    
    def __init__(self, config: TransformerConfig, num_classes: int, num_layers: int = 6):
        super().__init__()
        self.encoder = TransformerEncoder(config, num_layers)
        self.classifier = nn.Linear(config.d_model, num_classes)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            logits: [batch_size, num_classes]
        """
        encoder_output = self.encoder(input_ids, attention_mask)
        hidden_states = encoder_output['last_hidden_state']
        
        # Use [CLS] token representation (first token) for classification
        cls_representation = hidden_states[:, 0, :]  # [batch_size, d_model]
        cls_representation = self.dropout(cls_representation)
        
        logits = self.classifier(cls_representation)
        return logits


def attention_visualization_example():
    """Example demonstrating attention visualization and analysis."""
    print("=== Transformer Attention Mechanism Example ===")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    config = TransformerConfig(
        d_model=256,
        n_heads=8,
        d_ff=1024,
        max_seq_length=100,
        dropout=0.1,
        vocab_size=1000
    )
    
    print(f"Model Configuration:")
    print(f"  d_model: {config.d_model}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  d_ff: {config.d_ff}")
    print(f"  vocab_size: {config.vocab_size}")
    print()
    
    # Create model
    model = TransformerEncoder(config, num_layers=4)
    model.eval()
    
    # Sample input
    batch_size, seq_len = 2, 20
    input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len))
    
    # Add some padding tokens (0) to demonstrate masking
    input_ids[:, -5:] = 0  # Last 5 tokens are padding
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Sample input IDs: {input_ids[0][:10].tolist()}...")
    print()
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
    
    hidden_states = outputs['last_hidden_state']
    print(f"Output hidden states shape: {hidden_states.shape}")
    print(f"Hidden state statistics:")
    print(f"  Mean: {hidden_states.mean().item():.6f}")
    print(f"  Std: {hidden_states.std().item():.6f}")
    print(f"  Min: {hidden_states.min().item():.6f}")
    print(f"  Max: {hidden_states.max().item():.6f}")
    print()
    
    # Analyze attention patterns
    print("=== Attention Analysis ===")
    
    # Create a simple attention module for analysis
    attention_module = MultiHeadAttention(config.d_model, config.n_heads)
    attention_module.eval()
    
    with torch.no_grad():
        # Use the hidden states as Q, K, V
        sample_hidden = hidden_states[:1]  # Take first sample
        attn_output, attn_weights = attention_module(
            sample_hidden, sample_hidden, sample_hidden
        )
    
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Attention output shape: {attn_output.shape}")
    
    # Analyze attention patterns for each head
    attn_weights_np = attn_weights[0].numpy()  # [n_heads, seq_len, seq_len]
    
    print(f"\nAttention Pattern Analysis:")
    for head in range(min(4, config.n_heads)):  # Show first 4 heads
        head_weights = attn_weights_np[head]
        
        # Calculate attention entropy (how focused the attention is)
        entropy = -np.sum(head_weights * np.log(head_weights + 1e-10), axis=-1)
        avg_entropy = np.mean(entropy)
        
        # Calculate attention to self (diagonal elements)
        self_attention = np.mean(np.diag(head_weights))
        
        print(f"  Head {head}:")
        print(f"    Average entropy: {avg_entropy:.4f}")
        print(f"    Self-attention: {self_attention:.4f}")
        print(f"    Max attention: {head_weights.max():.4f}")
        print(f"    Min attention: {head_weights.min():.4f}")
    
    print()
    
    # Demonstrate positional encoding effects
    print("=== Positional Encoding Analysis ===")
    pos_encoding = PositionalEncoding(config.d_model, config.max_seq_length)
    
    # Create dummy input to get positional encodings
    dummy_input = torch.zeros(1, seq_len, config.d_model)
    pos_encoded = pos_encoding(dummy_input)
    pos_only = pos_encoded - dummy_input  # Extract just the positional encoding
    
    print(f"Positional encoding shape: {pos_only.shape}")
    print(f"Position encoding statistics:")
    print(f"  Mean: {pos_only.mean().item():.6f}")
    print(f"  Std: {pos_only.std().item():.6f}")
    
    # Analyze positional encoding patterns
    pos_np = pos_only[0].numpy()  # [seq_len, d_model]
    
    # Calculate similarity between different positions
    similarities = []
    for i in range(0, seq_len-1, 2):
        for j in range(i+1, seq_len, 2):
            similarity = np.dot(pos_np[i], pos_np[j]) / (
                np.linalg.norm(pos_np[i]) * np.linalg.norm(pos_np[j])
            )
            similarities.append(similarity)
    
    print(f"Average positional similarity: {np.mean(similarities):.4f}")
    print(f"Std positional similarity: {np.std(similarities):.4f}")
    print()
    
    # Classification example
    print("=== Classification Example ===")
    classifier = TransformerClassifier(config, num_classes=10, num_layers=2)
    classifier.eval()
    
    with torch.no_grad():
        logits = classifier(input_ids)
    
    predictions = torch.softmax(logits, dim=-1)
    print(f"Classification logits shape: {logits.shape}")
    print(f"Sample predictions (first sample): {predictions[0].tolist()}")
    print(f"Predicted class: {torch.argmax(predictions[0]).item()}")
    print()
    
    # Memory and computation analysis
    print("=== Computational Complexity Analysis ===")
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_params = count_parameters(model)
    classifier_params = count_parameters(classifier)
    
    print(f"Encoder parameters: {total_params:,}")
    print(f"Classifier parameters: {classifier_params:,}")
    
    # Theoretical complexity analysis
    n = seq_len
    d = config.d_model
    
    attention_complexity = config.n_heads * n * n * d  # O(n²d)
    ffn_complexity = n * d * config.d_ff * 2  # O(nd_ff)
    
    print(f"Attention complexity: O({attention_complexity:,}) operations")
    print(f"FFN complexity: O({ffn_complexity:,}) operations")
    print(f"Total per layer: O({attention_complexity + ffn_complexity:,}) operations")


if __name__ == "__main__":
    attention_visualization_example()