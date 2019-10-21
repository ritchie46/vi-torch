import torch
from torch import nn
import torch.distributions as dist
from layers.base import KL_Layer


class LinearVariational(KL_Layer):
    """
    Mean field approximation of nn.Linear
    """

    def __init__(self, in_features, out_features, n_batches, bias=True):
        """
        Parameters
        ----------
        in_features : int
        out_features : int
        n_batches : int
            Needed for KL-divergence re-scaling.
            See Also:
                Blundell, Cornebise, Kavukcioglu & Wierstra (2015, May 20)
                Weight Uncertainty in Neural Networks.
                Retrieved from https://arxiv.org/abs/1505.05424
        bias
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.include_bias = bias
        self.n_batches = n_batches

        # Initialize the variational parameters.
        # 𝑄(𝑤)=N(𝜇_𝜃,𝜎2_𝜃)
        # Do some random initialization with 𝜎=0.001
        self.w_mu = nn.Parameter(
            torch.FloatTensor(in_features, out_features).normal_(mean=0, std=0.001)
        )
        # proxy for variance
        # log(1 + exp(ρ))◦ eps
        self.w_p = nn.Parameter(
            torch.FloatTensor(in_features, out_features).normal_(mean=-2.5, std=0.001)
        )
        if self.include_bias:
            self.b_mu = nn.Parameter(torch.zeros(out_features))
            # proxy for variance
            self.b_p = nn.Parameter(torch.zeros(out_features))

    def reparameterize(self, mu, p):
        sigma = torch.log(1 + torch.exp(p))
        eps = torch.randn_like(sigma)
        return mu + (eps * sigma)

    def kl_divergence(self, z, mu_theta, p_theta, prior_sd=1):
        log_prior = dist.Normal(0, prior_sd).log_prob(z)
        log_p_q = dist.Normal(mu_theta, torch.log(1 + torch.exp(p_theta))).log_prob(z)
        return (log_p_q - log_prior).sum() / self.n_batches

    def forward(self, x):
        w = self.reparameterize(self.w_mu, self.w_p)

        if self.include_bias:
            b = self.reparameterize(self.b_mu, self.b_p)
        else:
            b = 0

        z = x @ w + b

        self._kl_divergence_ += self.kl_divergence(w, self.w_mu, self.w_p)
        if self.include_bias:
            self._kl_divergence_ += self.kl_divergence(b, self.b_mu, self.b_p)
        return z