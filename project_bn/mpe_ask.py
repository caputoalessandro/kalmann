from aima_probability import (
    BayesNet,
    pointwise_product,
)
from project_bn.maxout_factor import make_maxout_factor


def mpe_ask(e, bn: BayesNet):
    factors = [
        make_maxout_factor(var, e, bn) for var in reversed(bn.variables)
    ]
    to_max_out = [var for var in reversed(bn.variables) if var not in e]
    return maxout_all(factors, to_max_out, bn)


def max_out(var, factors, bn):
    """Eliminate var from all factors by summing over its values."""
    result, var_factors = [], []
    for f in factors:
        if var in f.variables:
            var_factors.append(f)
        else:
            result.append(f)
    result.append(pointwise_product(var_factors, bn).max_out(var, bn))
    return result


def maxout_all(factors, to_max_out, bn):

    for var in to_max_out:
        factors = max_out(var, factors, bn)

    factor = factors[0]
    return factor.previous_assignments[()], factor.cpt[()]
