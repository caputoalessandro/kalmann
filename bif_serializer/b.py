import timeit
from aima_probability import *
from project_bn.mpe_ask import mpe_ask
from project_bn.map_ask import map_ask
from bif_serializer import parse_bif_spec
from project_bn.nets.sprinkler_plus import make_sprinkler_plus

(
    ["Season", "Sprinkler"],
    dict(GrassWet=True, RoadWet=False),
    make_sprinkler_plus(),
    0.09923017499999999,
    dict(Season="summer", Sprinkler=True),
)

mpe_ask(
        dict(Alarm=True), BayesNet(parse_bif_spec("resources/earthquake.xml"))
    )

