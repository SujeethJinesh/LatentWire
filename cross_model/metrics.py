"""Metrics and loss functions for cross-model alignment."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCE(nn.Module):
    """
    InfoNCE loss for contrastive learning.

    References:
        - van den Oord et al., "Representation Learning with Contrastive Predictive Coding"
          arXiv:1807.03748 (2018)
        - Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations"
          arXiv:2002.05709 (SimCLR, 2020)
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negatives):
        """
        Args:
            anchor: [batch_size, hidden_dim]
            positive: [batch_size, hidden_dim]
            negatives: [batch_size, num_negatives, hidden_dim]
        """
        batch_size = anchor.shape[0]

        # Normalize representations
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        # Positive similarity
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature

        # Negative similarities
        neg_sim = torch.matmul(negatives, anchor.unsqueeze(-1)).squeeze(-1) / self.temperature

        # Compute InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)

        return F.cross_entropy(logits, labels)


class CKA:
    """
    Debiased CKA for measuring similarity between representations.

    Based on Murphy et al., ICLR 2024: "Unbiased HSIC Estimation"
    """

    @staticmethod
    def linear_kernel(X):
        """Linear kernel (dot product)."""
        return X @ X.T

    @staticmethod
    def center_gram(K):
        """Center gram matrix."""
        n = K.shape[0]
        H = torch.eye(n, device=K.device) - torch.ones((n, n), device=K.device) / n
        return H @ K @ H

    @staticmethod
    def unbiased_hsic_estimator(K, L):
        """
        Compute unbiased HSIC estimator (Murphy et al., ICLR 2024).

        Args:
            K: Gram matrix [n, n]
            L: Gram matrix [n, n]

        Returns:
            Unbiased HSIC estimate (scalar)
        """
        n = K.shape[0]

        if n < 4:
            # Fall back to standard HSIC for very small samples
            K_c = CKA.center_gram(K)
            L_c = CKA.center_gram(L)
            return torch.sum(K_c * L_c)

        # Remove diagonal (for unbiased estimation)
        K_tilde = K - torch.diag(torch.diag(K))
        L_tilde = L - torch.diag(torch.diag(L))

        # Unbiased HSIC formula
        trace_term = torch.trace(K_tilde @ L_tilde)
        sum_k = torch.sum(K_tilde)
        sum_l = torch.sum(L_tilde)
        sum_kl = torch.sum(K_tilde * L_tilde)

        hsic_unbiased = (
            trace_term +
            (sum_k * sum_l) / ((n - 1) * (n - 2)) -
            (2 * sum_kl) / (n - 2)
        ) / (n * (n - 3))

        return hsic_unbiased

    @staticmethod
    def cka_similarity(X, Y, debiased=True):
        """
        Compute CKA similarity between two representation matrices.

        Args:
            X, Y: [n_samples, n_features] representation matrices
            debiased: Use unbiased HSIC estimator (recommended for n < 1000)

        Returns:
            Scalar similarity score in [0, 1] (higher = more similar)

        References:
            - Kornblith et al., ICML 2019: "Similarity of Neural Network Representations Revisited"
            - Murphy et al., ICLR 2024: "Unbiased HSIC Estimation"
        """
        # Compute gram matrices
        K = CKA.linear_kernel(X)
        L = CKA.linear_kernel(Y)

        if debiased:
            # Use unbiased HSIC estimator
            hsic_xy = CKA.unbiased_hsic_estimator(K, L)
            hsic_xx = CKA.unbiased_hsic_estimator(K, K)
            hsic_yy = CKA.unbiased_hsic_estimator(L, L)

            # Prevent division by zero or negative values
            denominator = torch.sqrt(torch.clamp(hsic_xx * hsic_yy, min=1e-12))
            return hsic_xy / (denominator + 1e-8)
        else:
            # Standard CKA
            K_c = CKA.center_gram(K)
            L_c = CKA.center_gram(L)

            hsic = torch.sum(K_c * L_c)
            var_x = torch.sqrt(torch.sum(K_c * K_c))
            var_y = torch.sqrt(torch.sum(L_c * L_c))

            return hsic / (var_x * var_y + 1e-8)


class AlignmentUniformity:
    """
    Alignment and Uniformity metrics for contrastive learning quality.

    Based on Wang & Isola, ICML 2020: "Understanding Contrastive Representation Learning"
    """

    @staticmethod
    def alignment_loss(x, y, alpha=2):
        """
        Alignment: measures closeness of positive pairs.
        Lower is better (0 = perfect alignment).

        Args:
            x, y: [batch_size, hidden_dim] - positive pair representations
            alpha: distance power (default 2 for squared Euclidean distance)

        Returns:
            Alignment loss (scalar, lower = better alignment)
        """
        return (x - y).norm(dim=1).pow(alpha).mean()

    @staticmethod
    def uniformity_loss(x, t=2):
        """
        Uniformity: measures how uniformly representations are distributed.
        Lower is better (more uniform distribution prevents collapse).

        Args:
            x: [batch_size, hidden_dim] - representations
            t: temperature parameter (default 2)

        Returns:
            Uniformity loss (scalar, lower = more uniform distribution)
        """
        # Normalize representations to unit hypersphere
        x = F.normalize(x, dim=-1)

        # Compute pairwise squared distances
        sq_dist = torch.cdist(x, x).pow(2)

        # Log of mean of exponentials
        return torch.log(torch.exp(-t * sq_dist).mean() + 1e-8)

    @staticmethod
    def compute_metrics(anchor, positive):
        """
        Compute both alignment and uniformity metrics.

        Args:
            anchor, positive: [batch_size, hidden_dim] - positive pair representations

        Returns:
            (alignment_loss, uniformity_loss) - both scalars, lower is better
        """
        align = AlignmentUniformity.alignment_loss(anchor, positive)
        uniform = AlignmentUniformity.uniformity_loss(anchor)

        return align, uniform


def mean_pooling(token_embeddings, attention_mask):
    """
    Mean pooling with attention mask weighting.

    References:
        - Reimers & Gurevych, "Sentence-BERT" arXiv:1908.10084 (2019)
        - Muennighoff et al., "MTEB" arXiv:2210.07316 (2022)

    Args:
        token_embeddings: [batch_size, seq_len, hidden_dim]
        attention_mask: [batch_size, seq_len] (1 = real token, 0 = padding)

    Returns:
        pooled: [batch_size, hidden_dim] - mean of non-padded tokens
    """
    # Expand mask to match embedding dimensions
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    # Sum embeddings weighted by mask
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)

    # Sum mask values (number of real tokens)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    # Compute mean
    return sum_embeddings / sum_mask
