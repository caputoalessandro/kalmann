from aima_probability import BayesNet


def ptrue(p):
    return {True: p, False: 1 - p}


def complement(value, **kwargs):
    return {**kwargs, value: 1 - sum(kwargs.values())}


sprinkler_plus_spec = [
    (
        "Season",
        [],
        {(): {"spring": 0.25, "summer": 0.25, "fall": 0.25, "winter": 0.25}},
    ),
    ("Cloudy", [], {(): ptrue(0.4)}),
    (
        "Sprinkler",
        ["Season", "Cloudy"],
        {
            ("spring", True): ptrue(0.01),
            ("spring", False): ptrue(0.1),
            ("summer", True): ptrue(0.2),
            ("summer", False): ptrue(0.7),
            ("fall", True): ptrue(0.001),
            ("fall", False): ptrue(0.01),
            ("winter", True): ptrue(0),
            ("winter", False): ptrue(0),
        },
    ),
    (
        "Rain",
        ["Season", "Cloudy"],
        {
            ("spring", True): complement("no", heavy=0.2, light=0.4),
            ("spring", False): complement("no", heavy=0, light=0.1),
            ("summer", True): complement("no", heavy=0.6, light=0.15),
            ("summer", False): complement("no", heavy=0.01, light=0.04),
            ("fall", True): complement("no", heavy=0.3, light=0.5),
            ("fall", False): complement("no", heavy=0.001, light=0.2),
            ("winter", True): complement("no", heavy=0.01, light=0.79),
            ("winter", False): complement("no", heavy=0, light=0.01),
        },
    ),
    (
        "RoadWet",
        ["Rain"],
        {("heavy",): ptrue(1), ("light",): ptrue(0.9), ("no",): ptrue(0.01)},
    ),
    (
        "GrassWet",
        ["Sprinkler", "Rain"],
        {
            (True, "heavy"): ptrue(1),
            (True, "light"): ptrue(0.99),
            (True, "no"): ptrue(0.95),
            (False, "heavy"): ptrue(0.999),
            (False, "light"): ptrue(0.8),
            (False, "no"): ptrue(0.01),
        },
    ),
]


def make_sprinkler_plus():
    return BayesNet(sprinkler_plus_spec)
