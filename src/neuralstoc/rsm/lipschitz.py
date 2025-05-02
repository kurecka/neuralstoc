import numpy as np
import jax
import jax.numpy as jnp

from functools import partial

import torch
from auto_LiRPA.jacobian import JacobianOP, GradNorm
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


import logging
logger = logging.getLogger("neuralstoc")


def lipschitz_l1_jax(params, obs_normalization=None):
    """
    Computes the L1 Lipschitz constant of a JAX model.
    Optionally applies observation normalization if provided.
    """
    lipschitz_l1 = 1
    sum_axis = 1  # flax dense is transposed
    for i, (k, v) in enumerate(params["params"].items()):
        lipschitz_l1 *= jnp.max(jnp.sum(jnp.abs(v["kernel"]), axis=sum_axis))
    if obs_normalization is not None:
        lipschitz_l1 *= jnp.abs(1/obs_normalization.std).sum()

    return lipschitz_l1


def lipschitz_linf_jax(params, obs_normalization=None):
    """
    Computes the L-infinity Lipschitz constant of a JAX model.
    Optionally applies observation normalization if provided.
    """
    lipschitz_linf = 1
    sum_axis = 0  # flax dense is transposed
    for i, (k, v) in enumerate(params["params"].items()):
        lipschitz_linf *= jnp.max(jnp.sum(jnp.abs(v["kernel"]), axis=sum_axis))
    if obs_normalization is not None:
        lipschitz_linf *= jnp.max(1 / obs_normalization.std)
    return lipschitz_linf


@partial(jax.jit, static_argnums=(1, 2, 3,))
def lipschitz_coeff_CPLip(params, Linfty, weighted=True, CPLip=True, obs_normalization=None):
    '''
    Function to compute Lipschitz constants using the techniques presented in the paper.

    :param params: Neural network parameters.
    :param weighted: If true, use weighted norms.
    :param CPLip: If true, use the average activation operators (cplip) improvement.
    :param Linfty: If true, use Linfty norm; If false, use L1 norm (currently only L1 norm is used).
    :return: Lipschitz constant and list of weights (or None if weighted is False).
    '''

    if Linfty:
        axis = 0
    else:
        axis = 1

    minweight = jnp.float32(1e-6)
    maxweight = jnp.float32(1e6)

    if (not weighted and not CPLip):
        L = jnp.float32(1)
        # Compute Lipschitz coefficient by iterating through layers
        for layer in params["params"].values():
            # Involve only the 'kernel' dictionaries of each layer in the network, which are the weight matrices
            if "kernel" in layer:
                L *= jnp.max(jnp.sum(jnp.abs(layer["kernel"]), axis=axis))

    elif (not weighted and CPLip):
        L = jnp.float32(0)
        matrices = []
        for layer in params["params"].values():
            # Collect all weight matrices of the network
            if "kernel" in layer:
                matrices.append(layer["kernel"])

        nmatrices = len(matrices)
        # Create a list with all products of consecutive weight matrices
        # products[i][j] is the matrix product matrices[i + j] ... matrices[j]
        products = [matrices]
        prodnorms = [[jnp.max(jnp.sum(jnp.abs(mat), axis=axis)) for mat in matrices]]
        for nprods in range(1, nmatrices):
            prod_list = []
            for idx in range(nmatrices - nprods):
                prod_list.append(jnp.matmul(products[nprods - 1][idx], matrices[idx + nprods]))
            products.append(prod_list)
            prodnorms.append([jnp.max(jnp.sum(jnp.abs(mat), axis=axis)) for mat in prod_list])

        ncombs = 1 << (nmatrices - 1)
        for idx in range(ncombs):
            # To iterate over all possible ways of putting norms or products between the layers, 
            #  interpret idx as binary number of length (nmatrices - 1),
            # where the jth bit determines whether to put a norm or a product between layers j and j+1
            # We use that the (nmatrices - 1)th bit of such number is always 0, which implies that
            # each layer is taken into account for each term in the sum. 
            jprev = 0
            Lloc = jnp.float32(1)
            for jcur in range(nmatrices):
                if idx & (1 << jcur) == 0: 
                    Lloc *= prodnorms[jcur - jprev][jprev]
                    jprev = jcur + 1

            L += Lloc / ncombs


    elif (weighted and not CPLip and not Linfty):
        L = jnp.float32(1)
        matrices = []
        for layer in params["params"].values():
            # Collect all weight matrices of the network
            if "kernel" in layer:
                matrices.append(layer["kernel"])
        matrices.reverse()

        weights = [jnp.ones(jnp.shape(matrices[0])[1])]
        for mat in matrices:
            colsums = jnp.sum(jnp.multiply(jnp.abs(mat), weights[-1][jnp.newaxis, :]), axis=1)
            lip = jnp.maximum(jnp.max(colsums), minweight)
            weights.append(jnp.maximum(colsums / lip, minweight))
            L *= lip

    elif (weighted and not CPLip and Linfty):
        L = jnp.float32(1)
        matrices = []
        for layer in params["params"].values():
            # Collect all weight matrices of the network
            if "kernel" in layer:
                matrices.append(layer["kernel"])

        weights = [jnp.ones(jnp.shape(matrices[0])[0])]
        for mat in matrices:
            rowsums = jnp.sum(jnp.multiply(jnp.abs(mat), jnp.float32(1) / weights[-1][:, jnp.newaxis]), axis=0)
            lip = jnp.max(rowsums)
            weights.append(jnp.minimum(lip / rowsums, maxweight))
            L *= lip

    elif (weighted and CPLip and not Linfty):
        L = jnp.float32(0)
        matrices = []
        for layer in params["params"].values():
            # Collect all weight matrices of the network
            if "kernel" in layer:
                matrices.append(layer["kernel"])
        matrices.reverse()

        weights = [jnp.ones(jnp.shape(matrices[0])[1])]
        for mat in matrices:
            colsums = jnp.sum(jnp.multiply(jnp.abs(mat), weights[-1][jnp.newaxis, :]), axis=1)
            lip = jnp.maximum(jnp.max(colsums), minweight)
            weights.append(jnp.maximum(colsums / lip, minweight))

        matrices.reverse()
        nmatrices = len(matrices)
        # Create a list with all products of consecutive weight matrices
        # products[i][j] is the matrix product matrices[i + j] ... matrices[j]
        products = [matrices]
        prodnorms = [[jnp.max(jnp.multiply(jnp.sum(jnp.multiply(jnp.abs(matrices[idx]),
                                                                weights[-(idx + 2)][jnp.newaxis, :]), axis=1),
                                           jnp.float32(1) / weights[-(idx + 1)]))
                      for idx in range(nmatrices)]]
        for nprods in range(1, nmatrices):
            prod_list = []
            for idx in range(nmatrices - nprods):
                prod_list.append(jnp.matmul(products[nprods - 1][idx], matrices[idx + nprods]))
            products.append(prod_list)
            prodnorms.append([jnp.max(jnp.multiply(jnp.sum(jnp.multiply(jnp.abs(prod_list[idx]),
                                                                        weights[-(idx + nprods + 2)][jnp.newaxis, :]),
                                                           axis=1),
                                                   jnp.float32(1) / weights[-(idx + 1)]))
                              for idx in range(nmatrices - nprods)])

        ncombs = 1 << (nmatrices - 1)
        for idx in range(ncombs):
            # To iterate over all possible ways of putting norms or products between the layers, 
            #  interpret idx as binary number of length (nmatrices - 1),
            # where the jth bit determines whether to put a norm or a product between layers j and j+1
            # We use that the (nmatrices - 1)th bit of such number is always 0, which implies that
            # each layer is taken into account for each term in the sum. 
            jprev = 0
            Lloc = jnp.float32(1)
            for jcur in range(nmatrices):
                if idx & (1 << jcur) == 0: 
                    Lloc *= prodnorms[jcur - jprev][jprev]
                    jprev = jcur + 1

            L += Lloc / ncombs

    elif (weighted and CPLip and Linfty):
        L = jnp.float32(0)
        matrices = []
        for layer in params["params"].values():
            # Collect all weight matrices of the network
            if "kernel" in layer:
                matrices.append(layer["kernel"])

        weights = [jnp.ones(jnp.shape(matrices[0])[0])]
        for mat in matrices:
            rowsums = jnp.sum(jnp.multiply(jnp.abs(mat), jnp.float32(1) / weights[-1][:, jnp.newaxis]), axis=0)
            lip = jnp.max(rowsums)
            weights.append(jnp.minimum(lip / rowsums, maxweight))
        weights.reverse()

        nmatrices = len(matrices)
        # Create a list with all products of consecutive weight matrices
        # products[i][j] is the matrix product matrices[i + j] ... matrices[j]
        products = [matrices]
        prodnorms = [[jnp.max(jnp.multiply(jnp.sum(jnp.multiply(jnp.abs(matrices[idx]),
                                                                jnp.float32(1) / weights[-(idx + 1)][:, jnp.newaxis]),
                                                   axis=0),
                                           weights[-(idx + 2)]))
                      for idx in range(nmatrices)]]
        for nprods in range(1, nmatrices):
            prod_list = []
            for idx in range(nmatrices - nprods):
                prod_list.append(jnp.matmul(products[nprods - 1][idx], matrices[idx + nprods]))
            products.append(prod_list)
            prodnorms.append([jnp.max(jnp.multiply(jnp.sum(jnp.multiply(jnp.abs(prod_list[idx]),
                                                                        jnp.float32(1) / weights[-(idx + 1)][:,
                                                                                         jnp.newaxis]), axis=0),
                                                   weights[-(idx + nprods + 2)]))
                              for idx in range(nmatrices - nprods)])

        ncombs = 1 << (nmatrices - 1)
        for idx in range(ncombs):
            # To iterate over all possible ways of putting norms or products between the layers, 
            #  interpret idx as binary number of length (nmatrices - 1),
            # where the jth bit determines whether to put a norm or a product between layers j and j+1
            # We use that the (nmatrices - 1)th bit of such number is always 0, which implies that
            # each layer is taken into account for each term in the sum. 
            jprev = 0
            Lloc = jnp.float32(1)
            for jcur in range(nmatrices):
                if idx & (1 << jcur) == 0:
                    Lloc *= prodnorms[jcur - jprev][jprev]
                    jprev = jcur + 1

            L += Lloc / ncombs

        weights.reverse()

    if obs_normalization is not None:
        if Linfty:
            L *= jnp.max(1 / obs_normalization.std)
        else:
            L *= jnp.sum(1 / obs_normalization.std)

    if weighted:
        return L, weights[-1]
    else:
        return L, None


def compute_local_lipschitz(tnet, x0, eps, out_dim=1, obs_normalization=None):
    """
    Computes the local Lipschitz constant around a given point x0.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mean_ = None
    std_ = None
    if obs_normalization is not None:
        mean_ = np.asarray(obs_normalization.mean)
        std_ = np.asarray(obs_normalization.std)
        mean_ = torch.from_numpy(mean_).to(device).float()
        std_ = torch.from_numpy(std_).to(device).float()
    x0 = torch.from_numpy(x0).to(device).float()
    tnet = tnet.to(device)

    default_bound_opts = {
        'conv_mode': 'patches',
        'sparse_intermediate_bounds': False,
        'sparse_conv_intermediate_bounds': False,
        'sparse_intermediate_bounds_with_ibp': False,
        'sparse_features_alpha': False,
        'sparse_spec_alpha': False,
        'minimum_sparsity': 0.0,
        'enable_opt_interm_bounds': True,
        'crown_batch_size': np.inf,
        'forward_refinement': True,
        'dynamic_forward': True,
        'forward_max_dim': int(1e9),
        # Do not share alpha for conv layers.
        'use_full_conv_alpha': True,
        'disabled_optimization': [],
        # Threshold for number of unstable neurons for each layer to disable
        #  use_full_conv_alpha.
        'use_full_conv_alpha_thresh': 512,
        'verbosity': 0,
        'optimize_graph': {'optimizer': None},
    }

    class LocalLipschitzWrapper(torch.nn.Module):
        def __init__(self, model, mask):
            super().__init__()
            self.model = model
            self.mask = mask
            self.grad_norm = GradNorm(norm=1)

        def forward(self, x):
            if obs_normalization is not None:
                mean = mean_.to(x.device)
                std = std_.to(x.device)
                x = (x - mean) / std
            # y = self.model(x.permute(1, 0))
            y = self.model(x)
            self.mask = self.mask.to(y.device)
            y_selected = y.matmul(self.mask)
            jacobian = JacobianOP.apply(y_selected, x)
            lipschitz = self.grad_norm(jacobian)
            return lipschitz

    mask = torch.zeros(out_dim, 1)
    mask[0, 0] = 1
    x0.requires_grad_(True)
    sample = (x0[:1]).to(device)
    # # reorder x0 as pytorch batch shape
    # x0 = x0.permute(1, 0)
    # sample = sample.permute(1, 0)
    model = BoundedModule(LocalLipschitzWrapper(tnet, mask=mask), (sample), device=device, bound_opts=default_bound_opts)
    if obs_normalization is not None:
        mean_ = mean_.to(device)
        std_ = std_.to(device)
        sample_ = (sample - mean_) / std_
    else:
        sample_ = torch.tensor(sample).to(device)
    sample_.requires_grad_(True)
    y = tnet(sample_)
    ret_ori = torch.autograd.grad(y[:, 0].sum(), sample_)[0].abs().flatten(1).sum(dim=-1).view(-1)
    ret_new = model(sample, mask).view(-1)
    assert torch.allclose(ret_ori, ret_new)

    lip_ans = []
    for i in range(len(x0)):
        x = BoundedTensor(x0[i:i+1].to(device), PerturbationLpNorm(norm=np.inf, eps=eps))
        lip = []
        for j in range(mask.shape[0]):
            mask.zero_()
            mask[j, 0] = 1
            ub = model.compute_jacobian_bounds((x, mask), bound_lower=False)[1]
            lip.append(ub)
        lip = torch.max(torch.cat(lip))
        lip_ans.append(lip.item())

    return np.array(lip_ans)


def get_lipschitz_k(env, verifier, learner, log=None):
    """
    Compute the Lipschitz term for the verification.
    
    This function computes either global or local Lipschitz constants for both
    the controller and certificate networks, based on the selected norm (L1 or L-infinity).
    
    Returns:
        ndarray: The Lipschitz term(s) to be used in verification
    """
    if log is None:
        log = lambda **kwargs: None

    if verifier.norm == "l1":
        K_p = lipschitz_l1_jax(learner.p_state.params, obs_normalization=learner.obs_normalization).item()
        K_l = lipschitz_l1_jax(learner.v_state.params).item()
        K_f = env.lipschitz_constant
        lipschitz_k = K_l * K_f * (1 + K_p) + K_l

    else:
        if learner.model == "tmlp":
            learner.update_tmodels()

            grid, steps = verifier.get_unfiltered_grid_with_step()
            eps = 0.5 * np.sqrt(np.sum(steps ** 2))
            if learner.K_p is None:
                K_p = compute_local_lipschitz(learner.p_tnet, grid, eps, out_dim=learner.action_dim, obs_normalization=learner.obs_normalization)
            else:
                K_p = learner.K_p
            K_l = compute_local_lipschitz(learner.v_tnet, grid, eps)

            verifier.cached_lip_l_linf = jnp.float32(K_l)
            verifier.cached_lip_p_linf = jnp.float32(K_p)
            K_f = env.lipschitz_constant_linf
            lipschitz_k = K_l * K_f * np.maximum(1, K_p) + K_l

            global_K_l = np.max(K_l)
            global_K_p = np.max(K_p)
            learner.lip_lambda_l = np.max(K_l) / global_K_l
            learner.lip_lambda_p = np.max(K_p) / global_K_p

            log(
                eps=eps,
                K_l=global_K_l,
                K_p=global_K_p,
            )

        else:
            K_p = lipschitz_linf_jax(learner.p_state.params, obs_normalization=learner.obs_normalization).item()
            K_l = lipschitz_linf_jax(learner.v_state.params).item()
            K_f = env.lipschitz_constant_linf
            lipschitz_k = K_l * K_f * np.maximum(1, K_p) + K_l
            lipschitz_k = float(lipschitz_k)
            log(lipschitz_k=lipschitz_k)
            lipschitz_k = np.array([lipschitz_k])
            verifier.cached_lip_l_linf = np.array([jnp.float32(K_l)])
            verifier.cached_lip_p_linf = np.array([jnp.float32(K_p)])
            learner.lip_lambda_l = 1
            learner.lip_lambda_p = 1
            log(K_p=K_p)
            log(K_f=K_f)
            log(K_l=K_l)
    if verifier.norm != "linf":

        log(K_p=K_p)
        log(K_f=K_f)
        log(K_l=K_l)

        lipschitz_k = float(lipschitz_k)
        log(lipschitz_k=lipschitz_k)

    return lipschitz_k