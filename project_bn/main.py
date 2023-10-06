from aima_probability import *
from project_bn.mpe_ask import mpe_ask
from project_bn.map_ask import map_ask
from bif_serializer import parse_bif_spec
from project_bn.create_random_evidence import create_random_evidence
from project_bn.benchmark import benchmark

import os


def main():

    net = BayesNet(parse_bif_spec("resources/child.xml"))
    evidences, map_variables = create_random_evidence(
        len(net.nodes), net, 1, 10
    )
    # print(len(evidences), len(map_variables)), print(len(net.nodes))
    with benchmark("mpe") as mpe:
        ris_mpe = mpe_ask(evidences, net)

    with benchmark("map") as map:
        ris_map = map_ask(map_variables, evidences, net)

        print(ris_mpe)
        print("\n",ris_map)



if __name__ == "__main__":
    main()
