import pytest

from aima_probability import elimination_ask, enumeration_ask
from ..sprinkler_plus import make_sprinkler_plus

e = dict

queries = [
    ("GrassWet", e(Season="spring", Cloudy=True), "False: 0.472, True: 0.528"),
    (
        "Season",
        e(Rain="heavy"),
        "fall: 0.268, spring: 0.178, summer: 0.546, winter: 0.00888",
    ),
]

sprinkler_plus = make_sprinkler_plus()


@pytest.mark.parametrize("variable,evidence,expected", queries)
def test_sprinkler_enumeration_ask(variable, evidence, expected):
    result = enumeration_ask(variable, evidence, sprinkler_plus)
    print(result)
    assert result.show_approx() == expected


@pytest.mark.parametrize("variable,evidence,expected", queries)
def test_sprinkler_elimination_ask(variable, evidence, expected):
    result = elimination_ask(variable, evidence, sprinkler_plus)
    assert result.show_approx() == expected
