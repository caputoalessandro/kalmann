"""Probability models. (Chapter 13-15)
"""
import itertools
import random
from collections import defaultdict
from functools import reduce,singledispatch
from typing import Dict, List, Union

from .utils import (
    element_wise_product,
    first,
    inverse_matrix,
    isclose,
    matrix_multiplication,
    normalize,
    probability,
    product,
    scalar_vector_product,
    vector_add,
    vector_to_diagonal,
    weighted_sample_with_replacement,
)

Number = Union[int, float]


class ProbDist:
    """A discrete probability distribution. You name the random variable
    in the constructor, then assign and query probability of values.
    >>> P = ProbDist('Flip'); P['H'], P['T'] = 0.25, 0.75; P['H']
    0.25
    >>> P = ProbDist('X', {'lo': 125, 'med': 375, 'hi': 500})
    >>> P['lo'], P['med'], P['hi']
    (0.125, 0.375, 0.5)
    """

    def __init__(self, varname="?", freqs=None):
        """If freqs is given, it is a dictionary of values - frequency pairs,
        then ProbDist is normalized."""
        self.prob: Dict[str, Number] = {}
        self.varname: str = varname
        self.values: List[str] = []
        if freqs:
            for (v, p) in freqs.items():
                self[v] = p
            self.normalize()

    def __getitem__(self, val) -> Number:
        """Given a value, return P(value)."""
        try:
            return self.prob[val]
        except KeyError:
            return 0

    def __setitem__(self, val, p: Number):
        """Set P(val) = p."""
        if val not in self.values:
            self.values.append(val)
        self.prob[val] = p

    def normalize(self):
        """Make sure the probabilities of all values sum to 1.
        Returns the normalized distribution.
        Raises a ZeroDivisionError if the sum of the values is 0."""
        total = sum(self.prob.values())
        if not isclose(total, 1.0):
            for val in self.prob:
                self.prob[val] /= total
        return self

    def show_approx(self, numfmt="{:.3g}") -> str:
        """Show the probabilities rounded and sorted by key, for the
        sake of portable doctests."""
        return ", ".join(
            [
                ("{}: " + numfmt).format(v, p)
                for (v, p) in sorted(self.prob.items())
            ]
        )

    def __repr__(self):
        return "P({})".format(self.varname)


class JointProbDist(ProbDist):
    """A discrete probability distribute over a set of variables.
    >>> P = JointProbDist(['X', 'Y']); P[1, 1] = 0.25
    >>> P[1, 1]
    0.25
    >>> P[dict(X=0, Y=1)] = 0.5
    >>> P[dict(X=0, Y=1)]
    0.5"""

    def __init__(self, variables):
        self.prob = {}
        self.variables = variables
        self.vals = defaultdict(list)

    def __getitem__(self, values):
        """Given a tuple or dict of values, return P(values)."""
        values = event_values(values, self.variables)
        return ProbDist.__getitem__(self, values)

    def __setitem__(self, values, p):
        """Set P(values) = p.  Values can be a tuple or a dict; it must
        have a value for each of the variables in the joint. Also keep track
        of the values we have seen so far for each variable."""
        values = event_values(values, self.variables)
        self.prob[values] = p
        for var, val in zip(self.variables, values):
            if val not in self.vals[var]:
                self.vals[var].append(val)

    def values(self, var):
        """Return the set of possible values for a variable."""
        return self.vals[var]

    def __repr__(self):
        return "P({})".format(self.variables)


def event_values(event, variables):
    """Return a tuple of the values of variables in event.
    >>> event_values ({'A': 10, 'B': 9, 'C': 8}, ['C', 'A'])
    (8, 10)
    >>> event_values ((1, 2), ['C', 'A'])
    (1, 2)
    """
    if isinstance(event, tuple) and len(event) == len(variables):
        return event
    else:
        return tuple([event[var] for var in variables])


# ______________________________________________________________________________


def enumerate_joint_ask(X, e, P: JointProbDist):
    """Return a probability distribution over the values of the variable X,
    given the {var:val} observations e, in the JointProbDist P. [Section 13.3]
    >>> P = JointProbDist(['X', 'Y'])
    >>> P[0,0] = 0.25; P[0,1] = 0.5; P[1,1] = P[2,1] = 0.125
    >>> enumerate_joint_ask('X', dict(Y=1), P).show_approx()
    '0: 0.667, 1: 0.167, 2: 0.167'
    """
    assert X not in e, "Query variable must be distinct from evidence"
    Q = ProbDist(X)  # probability distribution for X, initially empty
    Y = [v for v in P.variables if v != X and v not in e]  # hidden variables.
    for xi in P.values(X):
        Q[xi] = enumerate_joint(Y, {**e, X: xi}, P)
    return Q.normalize()


def enumerate_joint(variables, e, P):
    """Return the sum of those entries in P consistent with e,
    provided variables is P's remaining variables (the ones not in e)."""
    if not variables:
        return P[e]
    Y, rest = variables[0], variables[1:]
    return sum([enumerate_joint(rest, {**e, Y: y}, P) for y in P.values(Y)])


# ______________________________________________________________________________


class BayesNet:
    """Bayesian network containing only boolean-variable nodes."""

    def __init__(self, node_specs=None):
        """Nodes must be ordered with parents before children."""
        self.nodes = []
        self.variables = []
        node_specs = node_specs or []
        for node_spec in node_specs:
            self.add(node_spec)

    def create_node(self, *args, **kwargs) -> "BayesNode":
        return BayesNode(*args, **kwargs)

    def add(self, node_spec):
        """Add a node to the net. Its parents must already be in the
        net, and its variable must not."""
        node = self.create_node(*node_spec)
        assert node.variable not in self.variables
        assert all((parent in self.variables) for parent in node.parents)
        self.nodes.append(node)
        self.variables.append(node.variable)
        for parent in node.parents:
            self.variable_node(parent).children.append(node)

    def variable_node(self, var) -> "BayesNode":
        """Return the node for the variable named var.
        >>> burglary.variable_node('Burglary').variable
        'Burglary'"""
        for n in self.nodes:
            if n.variable == var:
                return n
        raise Exception("No such variable: {}".format(var))

    def variable_values(self, var):
        """Return the domain of var."""
        return self.variable_node(var).values()

    def __repr__(self):
        return "BayesNet({0!r})".format(self.nodes)


class BoolBayesNet(BayesNet):
    """Bayesian network containing only boolean-variable nodes."""

    def create_node(self, *args, **kwargs) -> "BoolBayesNode":
        return BoolBayesNode(*args, **kwargs)


class BayesNode:
    """A conditional probability distribution for a variable,
    P(X | parents). Part of a BayesNet."""

    def __init__(self, X, parents, cpt):
        if isinstance(parents, str):
            parents = parents.split()

        self.variable = X
        self.parents = parents
        self.cpt = cpt
        self.children = []
        self.domain = sorted(self.determine_domain(cpt))

    @staticmethod
    def determine_domain(cpt: Dict):
        cpt_entry_keys = set(first(cpt.values()).keys())
        assert all(
            set(cpt_value.keys()) == cpt_entry_keys
            for cpt_value in cpt.values()
        )
        return cpt_entry_keys

    def p(self, value, event):
        """Return the conditional probability
        P(X=value | parents=parent_values), where parent_values
        are the values of parents in event. (event must assign each
        parent a value.)
        >>> bn = BoolBayesNode('X', 'Burglary', {T: 0.2, F: 0.625})
        >>> bn.p(False, {'Burglary': False, 'Earthquake': True})
        0.375"""
        return self.cpt[event_values(event, self.parents)][value]

    def values(self):
        return self.domain

    def __repr__(self):
        return repr((self.variable, " ".join(self.parents)))


class BoolBayesNode(BayesNode):
    """A conditional probability distribution for a boolean variable,
    P(X | parents). Part of a BayesNet."""

    def __init__(self, X, parents, cpt):
        """
        >>> X = BoolBayesNode('X', '', 0.2)
        >>> Y = BoolBayesNode('Y', 'P', {T: 0.2, F: 0.7})
        >>> Z = BoolBayesNode('Z', 'P Q',
        ...    {(T, T): 0.2, (T, F): 0.3, (F, T): 0.5, (F, F): 0.7})
        """
        if isinstance(parents, str):
            parents = parents.split()

        cpt = self._init_boolean_cpt(cpt, parents)
        super().__init__(X, parents, cpt)

    @staticmethod
    def _init_boolean_cpt(cpt, parents):

        if isinstance(cpt, (float, int)):  # no parents, 0-tuple
            cpt = {(): {True: cpt, False: 1 - cpt}}
        elif isinstance(cpt, dict):
            # one parent, 1-tuple
            if cpt and isinstance(list(cpt.keys())[0], bool):
                cpt = {(v,): {True: p, False: 1 - p} for v, p in cpt.items()}
            else:
                cpt = {v: {True: p, False: 1 - p} for v, p in cpt.items()}

        assert isinstance(cpt, dict)
        for vs, ps in cpt.items():
            assert isinstance(vs, tuple) and len(vs) == len(parents)
            assert all(isinstance(v, bool) for v in vs)
            for p in ps.values():
                assert 0 <= p <= 1

        return cpt

    def sample(self, event):
        """Sample from the distribution for this variable conditioned
        on event's values for parent_variables. That is, return True/False
        at random according with the conditional probability given the
        parents."""
        return probability(self.p(True, event))


# Burglary example [Figure 14.2]

T, F = True, False

burglary = BoolBayesNet(
    [
        ("Burglary", "", 0.001),
        ("Earthquake", "", 0.002),
        (
            "Alarm",
            "Burglary Earthquake",
            {(T, T): 0.95, (T, F): 0.94, (F, T): 0.29, (F, F): 0.001},
        ),
        ("JohnCalls", "Alarm", {T: 0.90, F: 0.05}),
        ("MaryCalls", "Alarm", {T: 0.70, F: 0.01}),
    ]
)

# ______________________________________________________________________________


def enumeration_ask(X, e, bn) -> ProbDist:
    """Return the conditional probability distribution of variable X
    given evidence e, from BayesNet bn. [Figure 14.9]
    >>> enumeration_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary
    ...  ).show_approx()
    'False: 0.716, True: 0.284'"""
    assert X not in e, "Query variable must be distinct from evidence"
    Q = ProbDist(X)
    for xi in bn.variable_values(X):
        Q[xi] = enumerate_all(bn.variables, {**e, X: xi}, bn)
    return Q.normalize()


def enumerate_all(variables, e, bn):
    """Return the sum of those entries in P(variables | e{others})
    consistent with e, where P is the joint distribution represented
    by bn, and e{others} means e restricted to bn's other variables
    (the ones other than variables). Parents must precede children in variables."""
    if not variables:
        return 1.0
    Y, rest = variables[0], variables[1:]
    Ynode = bn.variable_node(Y)
    if Y in e:
        return Ynode.p(e[Y], e) * enumerate_all(rest, e, bn)
    else:
        return sum(
            Ynode.p(y, e) * enumerate_all(rest, {**e, Y: y}, bn)
            for y in bn.variable_values(Y)
        )


# ______________________________________________________________________________


def elimination_ask(X, e, bn) -> ProbDist:
    """Compute bn's P(X|e) by variable elimination. [Figure 14.11]
    >>> elimination_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary
    ...  ).show_approx()
    'False: 0.716, True: 0.284'"""
    assert X not in e, "Query variable must be distinct from evidence"
    factors = []
    for var in reversed(bn.variables):
        factors.append(make_factor(var, e, bn))
        if is_hidden(var, X, e):
            factors = sum_out(var, factors, bn)
    return pointwise_product(factors, bn).normalize()


def is_hidden(var, X, e):
    """Is var a hidden variable when querying P(X|e)?"""
    if isinstance(X, list):
        return var not in X and var not in e
    else:
        return var != X and var not in e


def make_factor(var, e, bn: BayesNet) -> "Factor":
    """Return the factor for var in bn's joint distribution given e.
    That is, bn's full joint distribution, projected to accord with e,
    is the pointwise product of these factors for bn's variables."""
    node = bn.variable_node(var)
    variables = [X for X in [var] + node.parents if X not in e]
    cpt = {
        event_values(e1, variables): node.p(e1[var], e1)
        for e1 in all_events(variables, bn, e)
    }
    return Factor(variables, cpt)


def pointwise_product(factors, bn):
    return reduce(lambda f, g: f.pointwise_product(g, bn), factors)


def sum_out(var, factors, bn):
    """Eliminate var from all factors by summing over its values."""
    result, var_factors = [], []
    for f in factors:
        if var in f.variables:
            var_factors.append(f)
        else:
            result.append(f)
    result.append(pointwise_product(var_factors, bn).sum_out(var, bn))
    return result


class Factor:
    """A factor in a joint distribution."""

    def __init__(self, variables, cpt):
        self.variables = variables
        self.cpt = cpt

    def pointwise_product(self, other, bn) -> "Factor":
        """Multiply two factors, combining their variables."""
        variables = list(set(self.variables) | set(other.variables))
        cpt = {
            event_values(e, variables): self.p(e) * other.p(e)
            for e in all_events(variables, bn, {})
        }
        return Factor(variables, cpt)

    def sum_out(self, var, bn) -> "Factor":
        """Make a factor eliminating var by summing over its values."""
        variables = [X for X in self.variables if X != var]
        cpt = {
            event_values(e, variables): sum(
                self.p({**e, var: val}) for val in bn.variable_values(var)
            )
            for e in all_events(variables, bn, {})
        }
        return Factor(variables, cpt)

    def normalize(self) -> ProbDist:
        """Return my probabilities; must be down to one variable."""
        assert len(self.variables) == 1
        return ProbDist(
            self.variables[0], {k: v for ((k,), v) in self.cpt.items()}
        )

    def p(self, e):
        """Look up my value tabulated for e."""
        return self.cpt[event_values(e, self.variables)]


def all_events(variables, bn, e):
    """Yield every way of extending e with values for all variables."""
    possible_values = [bn.variable_values(var) for var in variables]

    for values in itertools.product(*possible_values):
        result = dict(zip(variables, values))
        result.update(e)
        yield result


# ______________________________________________________________________________

# [Figure 14.12a]: sprinkler network


sprinkler = BoolBayesNet(
    [
        ("Cloudy", "", 0.5),
        ("Sprinkler", "Cloudy", {T: 0.10, F: 0.50}),
        ("Rain", "Cloudy", {T: 0.80, F: 0.20}),
        (
            "WetGrass",
            "Sprinkler Rain",
            {(T, T): 0.99, (T, F): 0.90, (F, T): 0.90, (F, F): 0.00},
        ),
    ]
)

# ______________________________________________________________________________


def prior_sample(bn):
    """Randomly sample from bn's full joint distribution. The result
    is a {variable: value} dict. [Figure 14.13]"""
    event = {}
    for node in bn.nodes:
        event[node.variable] = node.sample(event)
    return event


# _________________________________________________________________________


def rejection_sampling(X, e, bn, N=10000):
    """Estimate the probability distribution of variable X given
    evidence e in BayesNet bn, using N samples.  [Figure 14.14]
    Raises a ZeroDivisionError if all the N samples are rejected,
    i.e., inconsistent with e.
    >>> random.seed(47)
    >>> rejection_sampling('Burglary', dict(JohnCalls=T, MaryCalls=T),
    ...   burglary, 10000).show_approx()
    'False: 0.7, True: 0.3'
    """
    counts = {x: 0 for x in bn.variable_values(X)}  # bold N in [Figure 14.14]
    for j in range(N):
        sample = prior_sample(bn)  # boldface x in [Figure 14.14]
        if consistent_with(sample, e):
            counts[sample[X]] += 1
    return ProbDist(X, counts)


def consistent_with(event, evidence):
    """Is event consistent with the given evidence?"""
    return all(evidence.get(k, v) == v for k, v in event.items())


# _________________________________________________________________________


def likelihood_weighting(X, e, bn, N=10000):
    """Estimate the probability distribution of variable X given
    evidence e in BayesNet bn.  [Figure 14.15]
    >>> random.seed(1017)
    >>> likelihood_weighting('Burglary', dict(JohnCalls=T, MaryCalls=T),
    ...   burglary, 10000).show_approx()
    'False: 0.702, True: 0.298'
    """
    W = {x: 0 for x in bn.variable_values(X)}
    for j in range(N):
        sample, weight = weighted_sample(
            bn, e
        )  # boldface x, w in [Figure 14.15]
        W[sample[X]] += weight
    return ProbDist(X, W)


def weighted_sample(bn, e):
    """Sample an event from bn that's consistent with the evidence e;
    return the event and its weight, the likelihood that the event
    accords to the evidence."""
    w = 1
    event = dict(e)  # boldface x in [Figure 14.15]
    for node in bn.nodes:
        Xi = node.variable
        if Xi in e:
            w *= node.p(e[Xi], event)
        else:
            event[Xi] = node.sample(event)
    return event, w


# _________________________________________________________________________


def gibbs_ask(X, e, bn, N=1000):
    """[Figure 14.16]"""
    assert X not in e, "Query variable must be distinct from evidence"
    counts = {x: 0 for x in bn.variable_values(X)}  # bold N in [Figure 14.16]
    Z = [var for var in bn.variables if var not in e]
    state = dict(e)  # boldface x in [Figure 14.16]
    for Zi in Z:
        state[Zi] = random.choice(bn.variable_values(Zi))
    for j in range(N):
        for Zi in Z:
            state[Zi] = markov_blanket_sample(Zi, state, bn)
            counts[state[X]] += 1
    return ProbDist(X, counts)


def markov_blanket_sample(X, e, bn):
    """Return a sample from P(X | mb) where mb denotes that the
    variables in the Markov blanket of X take their values from event
    e (which must assign a value to each). The Markov blanket of X is
    X's parents, children, and children's parents."""
    Xnode = bn.variable_node(X)
    Q = ProbDist(X)
    for xi in bn.variable_values(X):
        ei = {**e, X: xi}
        # [Equation 14.12:]
        Q[xi] = Xnode.p(xi, e) * product(
            Yj.p(ei[Yj.variable], ei) for Yj in Xnode.children
        )
    # (assuming a Boolean variable here)
    return probability(Q.normalize()[True])


# _________________________________________________________________________


class HiddenMarkovModel:
    """A Hidden markov model which takes Transition model and Sensor model as inputs"""

    def __init__(self, transition_model, sensor_model, prior=None):
        self.transition_model = transition_model
        self.sensor_model = sensor_model
        self.prior = prior or [0.5, 0.5]

    def sensor_dist(self, ev):
        if ev is True:
            return self.sensor_model[0]
        else:
            return self.sensor_model[1]


def forward(HMM, fv, ev):
    prediction = vector_add(
        scalar_vector_product(fv[0], HMM.transition_model[0]),
        scalar_vector_product(fv[1], HMM.transition_model[1]),
    )
    sensor_dist = HMM.sensor_dist(ev)

    return normalize(element_wise_product(sensor_dist, prediction))


def backward(HMM, b, ev):
    sensor_dist = HMM.sensor_dist(ev)
    prediction = element_wise_product(sensor_dist, b)

    return normalize(
        vector_add(
            scalar_vector_product(prediction[0], HMM.transition_model[0]),
            scalar_vector_product(prediction[1], HMM.transition_model[1]),
        )
    )


def forward_backward(HMM, ev, prior):
    """[Figure 15.4]
    Forward-Backward algorithm for smoothing. Computes posterior probabilities
    of a sequence of states given a sequence of observations."""
    t = len(ev)
    ev.insert(0, None)  # to make the code look similar to pseudo code

    fv = [[0.0, 0.0] for _ in range(len(ev))]
    b = [1.0, 1.0]
    bv = [
        b
    ]  # we don't need bv; but we will have a list of all backward messages here
    sv = [[0, 0] for _ in range(len(ev))]

    fv[0] = prior

    for i in range(1, t + 1):
        fv[i] = forward(HMM, fv[i - 1], ev[i])
    for i in range(t, -1, -1):
        sv[i - 1] = normalize(element_wise_product(fv[i], b))
        b = backward(HMM, b, ev[i])
        bv.append(b)

    sv = sv[::-1]

    return sv


# _________________________________________________________________________


def fixed_lag_smoothing(e_t, HMM, d, ev, t):
    """[Figure 15.6]
    Smoothing algorithm with a fixed time lag of 'd' steps.
    Online algorithm that outputs the new smoothed estimate if observation
    for new time step is given."""
    ev.insert(0, None)

    T_model = HMM.transition_model
    f = HMM.prior
    B = [[1, 0], [0, 1]]
    evidence = []

    evidence.append(e_t)
    O_t = vector_to_diagonal(HMM.sensor_dist(e_t))
    if t > d:
        f = forward(HMM, f, e_t)
        O_tmd = vector_to_diagonal(HMM.sensor_dist(ev[t - d]))
        B = matrix_multiplication(
            inverse_matrix(O_tmd), inverse_matrix(T_model), B, T_model, O_t
        )
    else:
        B = matrix_multiplication(B, T_model, O_t)
    t += 1

    if t > d:
        # always returns a 1x2 matrix
        return [normalize(i) for i in matrix_multiplication([f], B)][0]
    else:
        return None


# _________________________________________________________________________


def particle_filtering(e, N, HMM):
    """Particle filtering considering two states variables."""
    dist = [0.5, 0.5]
    # Weight Initialization
    w = [0 for _ in range(N)]
    # STEP 1
    # Propagate one step using transition model given prior state
    dist = vector_add(
        scalar_vector_product(dist[0], HMM.transition_model[0]),
        scalar_vector_product(dist[1], HMM.transition_model[1]),
    )
    # Assign state according to probability
    s = ["A" if probability(dist[0]) else "B" for _ in range(N)]
    w_tot = 0
    # Calculate importance weight given evidence e
    for i in range(N):
        if s[i] == "A":
            # P(U|A)*P(A)
            w_i = HMM.sensor_dist(e)[0] * dist[0]
        if s[i] == "B":
            # P(U|B)*P(B)
            w_i = HMM.sensor_dist(e)[1] * dist[1]
        w[i] = w_i
        w_tot += w_i

    # Normalize all the weights
    for i in range(N):
        w[i] = w[i] / w_tot

    # Limit weights to 4 digits
    for i in range(N):
        w[i] = float("{0:.4f}".format(w[i]))

    # STEP 2

    s = weighted_sample_with_replacement(N, s, w)

    return s


# _________________________________________________________________________
## TODO: Implement continuous map for MonteCarlo similar to Fig25.10 from the book


class MCLmap:
    """Map which provides probability distributions and sensor readings.
    Consists of discrete cells which are either an obstacle or empty"""

    def __init__(self, m):
        self.m = m
        self.nrows = len(m)
        self.ncols = len(m[0])
        # list of empty spaces in the map
        self.empty = [
            (i, j)
            for i in range(self.nrows)
            for j in range(self.ncols)
            if not m[i][j]
        ]

    def sample(self):
        """Returns a random kinematic state possible in the map"""
        pos = random.choice(self.empty)
        # 0N 1E 2S 3W
        orient = random.choice(range(4))
        kin_state = pos + (orient,)
        return kin_state

    def ray_cast(self, sensor_num, kin_state):
        """Returns distace to nearest obstacle or map boundary in the direction of sensor"""
        pos = kin_state[:2]
        orient = kin_state[2]
        # sensor layout when orientation is 0 (towards North)
        #  0
        # 3R1
        #  2
        delta = (
            (sensor_num % 2 == 0) * (sensor_num - 1),
            (sensor_num % 2 == 1) * (2 - sensor_num),
        )
        # sensor direction changes based on orientation
        for _ in range(orient):
            delta = (delta[1], -delta[0])
        range_count = 0
        while (
            (0 <= pos[0] < self.nrows)
            and (0 <= pos[1] < self.nrows)
            and (not self.m[pos[0]][pos[1]])
        ):
            pos = vector_add(pos, delta)
            range_count += 1
        return range_count


def monte_carlo_localization(a, z, N, P_motion_sample, P_sensor, m, S=None):
    """Monte Carlo localization algorithm from Fig 25.9"""

    def ray_cast(sensor_num, kin_state, m):
        return m.ray_cast(sensor_num, kin_state)

    M = len(z)
    W = [0] * N
    S_ = [0] * N
    W_ = [0] * N
    v = a["v"]
    w = a["w"]

    if S is None:
        S = [m.sample() for _ in range(N)]

    for i in range(N):
        S_[i] = P_motion_sample(S[i], v, w)
        W_[i] = 1
        for j in range(M):
            z_ = ray_cast(j, S_[i], m)
            W_[i] = W_[i] * P_sensor(z[j], z_)

    S = weighted_sample_with_replacement(N, S_, W_)
    return S
