from math import isclose

import pytest

from bif_serializer.deserialize import parse_bif_spec
from project_bn.nets.sprinkler_plus import sprinkler_plus_spec


def boolean_domain_mapper(_, domain):
    return [
        {"true": True, "false": False}.get(var.lower(), var) for var in domain
    ]


@pytest.mark.parametrize(
    "filepath,expected", [("resources/SprinklerPlus.xml", sprinkler_plus_spec)]
)
def test_deserialize(filepath, expected):
    spec = parse_bif_spec(filepath, domain_mapper=boolean_domain_mapper)
    specs_eq(spec, sprinkler_plus_spec)


def test_topological_sort():
    spec = parse_bif_spec(
        "resources/SprinklerPlus.xml", domain_mapper=boolean_domain_mapper
    )

    visited = set()

    for var, parents, *_ in spec:
        visited.add(var)
        assert all(parent in visited for parent in parents)


def specs_eq(spec1, spec2):
    spec1 = sorted(spec1, key=lambda e: e[0])
    spec2 = sorted(spec2, key=lambda e: e[0])

    for (name1, parents1, cpt1), (name2, parents2, cpt2) in zip(spec1, spec2):
        assert name1 == name2
        assert parents1 == parents2
        assert len(cpt1) == len(cpt2)

        for key, cptvalues1 in cpt1.items():
            cptvalues2 = cpt2[key]

            for var_value, p1 in cptvalues1.items():
                p2 = cptvalues2[var_value]

                assert isclose(p1, p2)
