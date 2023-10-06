from math import isclose

import pytest

from bif_serializer.deserialize import parse_bif
from project_bn.map_ask import map_ask
from project_bn.mpe_ask import mpe_ask
from project_bn.nets import make_sprinkler_plus


@pytest.mark.parametrize(
    "Ms,e,bn,expected_map_value,expected_assignments",
    [
        (
            ["Season", "Sprinkler"],
            dict(GrassWet=True, RoadWet=False),
            make_sprinkler_plus(),
            0.09923017499999999,
            dict(Season="summer", Sprinkler=True),
        ),
        (
            ["Burglary", "Earthquake", "JohnCalls", "MaryCalls"],
            {},
            parse_bif("resources/earthquake.xml"),
            0.9115897329,
            dict(
                Burglary=False,
                Earthquake=False,
                JohnCalls=False,
                MaryCalls=False,
            ),
        ),
    ],
)
def test_map(Ms, e, bn, expected_map_value, expected_assignments):
    assignments, map_value = map_ask(Ms, e, bn)
    assert assignments == expected_assignments
    assert isclose(map_value, expected_map_value)


@pytest.mark.parametrize(
    "e,bn,expected_mpe_value,expected_assignments",
    [
        (
            dict(GrassWet=True, RoadWet=False),
            make_sprinkler_plus(),
            0.09381487499999999,
            dict(Season="summer", Sprinkler=True, Cloudy=False, Rain="no"),
        )
    ],
)
def test_mpe(e, bn, expected_mpe_value, expected_assignments):
    assignments, mpe_value = mpe_ask(e, bn)
    assert assignments == expected_assignments
    assert isclose(mpe_value, expected_mpe_value)
