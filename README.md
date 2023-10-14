This repository contains an implementation of a simple **Gaussian mixture model** (GMM) fitted with Expectation-Maximization in [pytorch](http://www.pytorch.org). The interface closely follows that of [sklearn](http://scikit-learn.org).

![Example of a fit via a Gaussian Mixture model.](example.png)

---

A new model is instantiated by calling `gmm.GaussianMixture(..)` and providing as arguments the number of components, as well as the tensor dimension. Note that once instantiated, the model expects tensors in a flattened shape `(n, d)`.

The first step would usually be to fit the model via `model.fit(data)`, then predict with `model.predict(data)`. To reproduce the above figure, just run the provided `example.py`.

Some sanity checks can be executed by calling `python test.py`. To fit data on GPUs, ensure that you first call `model.cuda()`.

## Changes
Some parts of code are changed, including
1. In `self._m_step`, when `covariance_type == 'full'`, it should be `var = ... / (... + self.eps) + eps` in order to prevent division by zero.
2. In `self._estimate_log_prob`, when `covariance_type == 'diag'`, it should be `log_p = torch.sum((mu * mu + x * x - 2 * x * mu) * prec ** 2, dim=2, keepdim=True)`.
3. In `self._init_params`, use `.copy_` to initialize parameters instead of new a `nn.Parameter`.
4. In original implementation, `self._estimate_log_prob` is called twice in each iteration. This implementation improved it.
5. In `self.get_kmeans_mu`, the inerative random initialization is changed to kmeans++ initialization.
