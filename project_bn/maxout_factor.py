from aima_probability import (
    BayesNet,
    Factor,
    all_events,
    event_values,
    make_factor,
)

__all__ = ["MaxoutFactor", "make_maxout_factor"]


class MaxoutFactor:
    """A factor in a joint distribution."""

    def __init__(self, factor: Factor, previous_assignments=None):
        self.factor = factor
        self.previous_assignments = previous_assignments or {
            key: {} for key in factor.cpt.keys()
        }

    def pointwise_product(self, other: "MaxoutFactor", bn) -> "MaxoutFactor":
        """Multiply two factors, combining their variables."""
        factor = self.factor.pointwise_product(other, bn)
        assignments = {
            event_values(e, factor.variables): {
                **self.previous_assignments[event_values(e, self.variables)],
                **other.previous_assignments[event_values(e, other.variables)],
            }
            for e in all_events(factor.variables, bn, {})
        }
        return MaxoutFactor(factor, assignments)

    def max_out(self, var, bn) -> "MaxoutFactor":
        """Make a factor eliminating var by summing over its values."""
        variables = [X for X in self.variables if X != var]
        cpt = {}
        mpe = {}

        for e in all_events(variables, bn, {}):
            cpt_key = event_values(e, variables)
            max_p = -1
            max_value = None
            for val in bn.variable_values(var):
                val_p = self.p({**e, var: val})
                if max_p < val_p:
                    max_value = val
                    max_p = val_p

            cpt[cpt_key] = max_p
            mpe[cpt_key] = {
                **self.previous_assignments[
                    event_values({**e, var: max_value}, self.variables)
                ],
                var: max_value,
            }

        return MaxoutFactor(Factor(variables, cpt), mpe)

    def __getattr__(self, item):
        return getattr(self.factor, item)


def make_maxout_factor(var, e, bn: BayesNet) -> "MaxoutFactor":
    return MaxoutFactor(make_factor(var, e, bn))
