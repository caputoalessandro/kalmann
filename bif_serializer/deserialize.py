import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Iterable
import xml.etree.ElementTree as element_tree

from more_itertools import chunked

from aima_probability import BayesNet


@dataclass
class Variable:
    name: str
    domain: List[str]
    parents: List["Variable"] = field(init=False)
    children: List["Variable"] = field(default_factory=list, init=False)
    cpt: Dict = field(init=False)

    def set_cpt(self, probs: List[float]):
        parents_domains = [parent.domain for parent in self.parents]
        all_parents_assignments = itertools.product(*parents_domains)
        all_probs_given_parents = chunked(probs, len(self.domain))

        cpt = {}
        for parents_assignment, probs_given_parents in zip(
            all_parents_assignments, all_probs_given_parents
        ):
            cpt_given_parents = dict(zip(self.domain, probs_given_parents))
            cpt[parents_assignment] = cpt_given_parents

        self.cpt = cpt

    def to_bayes_node_spec(self):
        return self.name, [parent.name for parent in self.parents], self.cpt


def topological_sort(variables: Iterable[Variable]):
    sorted_variables = []
    temp_marked = set()
    perm_marked = set()

    def visit(var: Variable):
        if var.name in perm_marked:
            return
        if var.name in temp_marked:
            raise ValueError("The network is not a DAG.")

        temp_marked.add(var.name)

        for child in var.children:
            visit(child)

        temp_marked.remove(var.name)
        perm_marked.add(var.name)
        sorted_variables.append(var)

    for var in variables:
        visit(var)

    sorted_variables.reverse()
    return sorted_variables


def boolean_domain_mapper(_, domain):
    if domain == ["True", "False"]:
        return [True, False]
    if domain == ["False", "True"]:
        return [False, True]
    return domain


def parse_bif_spec(path: str, domain_mapper=boolean_domain_mapper):
    tree = element_tree.parse(path)
    root = tree.getroot().find("NETWORK")
    variables: Dict[str, Variable] = {}

    for vartag in root.findall("VARIABLE"):
        name = vartag.find("NAME").text
        domain = [outcometag.text for outcometag in vartag.findall("OUTCOME")]

        if domain_mapper:
            domain = domain_mapper(name, domain)

        var = Variable(name, domain)
        variables[name] = var

    for deftag in root.findall("DEFINITION"):
        var = variables[deftag.find("FOR").text]
        var.parents = [
            variables[giventag.text] for giventag in deftag.findall("GIVEN")
        ]
        probs = [float(s) for s in deftag.find("TABLE").text.split()]
        var.set_cpt(probs)

        for parent in var.parents:
            parent.children.append(var)

    return [
        var.to_bayes_node_spec()
        for var in topological_sort(variables.values())
    ]


def parse_bif(*args, **kwargs) -> BayesNet:
    return BayesNet(parse_bif_spec(*args, **kwargs))
