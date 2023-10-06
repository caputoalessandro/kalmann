from typing import List, Tuple

from aima_probability import BayesNet, is_hidden, make_factor, sum_out
from project_bn.maxout_factor import MaxoutFactor
from project_bn.mpe_ask import maxout_all


def map_ask(Ms: List[str], e: dict, bn: BayesNet) -> Tuple[dict, float]:
    factors = []
    for var in reversed(bn.variables):
        factors.append(make_factor(var, e, bn))
        if is_hidden(var, Ms, e):
            factors = sum_out(var, factors, bn)

    return maxout_all([MaxoutFactor(f) for f in factors], Ms, bn)
