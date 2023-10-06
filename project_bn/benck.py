from aima_probability import *
from project_bn.mpe_ask import mpe_ask
from project_bn.map_ask import map_ask
from bif_serializer import parse_bif_spec
from project_bn.create_random_evidence import create_random_evidence
from project_bn.benchmark import benchmark

import os


def main():

    ris1 = 0
    ris2 = 0

    print(
        "-------------------------------------------------------------------------------------"
    )
    print("CHILD - DOMINIO PICCOLO - EVIDENZE 10% MAP 25% ")

    for i in range(100):

        net = BayesNet(parse_bif_spec("resources/child.xml"))
        evidences, map_variables = create_random_evidence(
            len(net.nodes), net, 40, 10
        )
        # print(len(evidences), len(map_variables)), print(len(net.nodes))
        with benchmark("") as mpe:
            ris_mpe = mpe_ask(evidences, net)

        with benchmark("") as map:
            ris_map = map_ask(map_variables, evidences, net)

        ris1 = (mpe.duration_s + ris1) / 2
        ris2 = (map.duration_s + ris2) / 2

        if i == 99:
            print("mpe = ", ris1, "\nmap = ", ris2)

    print(
        "-------------------------------------------------------------------------------------"
    )

    print(
        "-------------------------------------------------------------------------------------"
    )
    print("CHILD - DOMINIO PICCOLO EVIDENZE 10% MAP 50% ")

    for i in range(100):

        net = BayesNet(parse_bif_spec("resources/child.xml"))
        evidences, map_variables = create_random_evidence(
            len(net.nodes), net, 40, 25
        )
        # print(len(evidences), len(map_variables)), print(len(net.nodes))
        with benchmark("") as mpe:
            ris_mpe = mpe_ask(evidences, net)

        with benchmark("") as map:
            ris_map = map_ask(map_variables, evidences, net)

        ris1 = (mpe.duration_s + ris1) / 2
        ris2 = (map.duration_s + ris2) / 2

        if i == 99:
            print("mpe = ", ris1, "\nmap = ", ris2)

    print(
        "-------------------------------------------------------------------------------------"
    )
    print("CHILD - DOMINIO PICCOLO 10% evidenze 75% MAP")

    for i in range(100):

        net = BayesNet(parse_bif_spec("resources/child.xml"))
        evidences, map_variables = create_random_evidence(
            len(net.nodes), net, 10, 50
        )
        # print(len(evidences), len(map_variables)), print(len(net.nodes))
        with benchmark("") as mpe:
            ris_mpe = mpe_ask(evidences, net)

        with benchmark("") as map:
            ris_map = map_ask(map_variables, evidences, net)

        ris1 = (mpe.duration_s + ris1) / 2
        ris2 = (map.duration_s + ris2) / 2

        if i == 99:
            print("mpe = ", ris1, "\nmap = ", ris2)

    print(
        "-------------------------------------------------------------------------------------"
    )
    print("CHILD - DOMINIO PICCOLO 10% evidenze 40% map")

    for i in range(100):

        net = BayesNet(parse_bif_spec("resources/child.xml"))
        evidences, map_variables = create_random_evidence(
            len(net.nodes), net, 10, 40
        )
        # print(len(evidences), len(map_variables)), print(len(net.nodes))
        with benchmark("") as mpe:
            ris_mpe = mpe_ask(evidences, net)

        with benchmark("") as map:
            ris_map = map_ask(map_variables, evidences, net)

        ris1 = (mpe.duration_s + ris1) / 2
        ris2 = (map.duration_s + ris2) / 2

        if i == 99:
            print("mpe = ", ris1, "\nmap = ", ris2)

    print(
        "-------------------------------------------------------------------------------------"
    )
    print("CHILD - DOMINIO PICCOLO 25% evidenze 40% map")

    for i in range(100):

        net = BayesNet(parse_bif_spec("resources/child.xml"))
        evidences, map_variables = create_random_evidence(
            len(net.nodes), net, 25, 40
        )
        # print(len(evidences), len(map_variables)), print(len(net.nodes))
        with benchmark("") as mpe:
            ris_mpe = mpe_ask(evidences, net)

        with benchmark("") as map:
            ris_map = map_ask(map_variables, evidences, net)

        ris1 = (mpe.duration_s + ris1) / 2
        ris2 = (map.duration_s + ris2) / 2

        if i == 99:
            print("mpe = ", ris1, "\nmap = ", ris2)

    print(
        "-------------------------------------------------------------------------------------"
    )
    print("CHILD - DOMINIO PICCOLO 75% evidenze 40% map")

    for i in range(100):

        net = BayesNet(parse_bif_spec("resources/child.xml"))
        evidences, map_variables = create_random_evidence(
            len(net.nodes), net, 50, 40
        )
        # print(len(evidences), len(map_variables)), print(len(net.nodes))
        with benchmark("") as mpe:
            ris_mpe = mpe_ask(evidences, net)

        with benchmark("") as map:
            ris_map = map_ask(map_variables, evidences, net)

        ris1 = (mpe.duration_s + ris1) / 2
        ris2 = (map.duration_s + ris2) / 2

        if i == 99:
            print("mpe = ", ris1, "\nmap = ", ris2)

    print(
        "-------------------------------------------------------------------------------------"
    )
    print("INSURANCE DIVERSA COMPLESSITÀ 10% evidenze - 50% map")
    for i in range(10):

        net = BayesNet(parse_bif_spec("resources/insurance.xml"))
        evidences, map_variables = create_random_evidence(
            len(net.nodes), net, 10, 50
        )
        # print(len(evidences), len(map_variables)), print(len(net.nodes))
        with benchmark("") as mpe:
            ris_mpe = mpe_ask(evidences, net)

        with benchmark("") as map:
            ris_map = map_ask(map_variables, evidences, net)

        ris1 = (mpe.duration_s + ris1) / 2
        ris2 = (map.duration_s + ris2) / 2

        if i == 9:
            print("mpe = ", ris1, "\nmap = ", ris2)

    print(
        "-------------------------------------------------------------------------------------"
    )

    print(
        "-------------------------------------------------------------------------------------"
    )
    print("HAILFINDER - DOMINIO MEDIO  10% evidenze - 50% map")
    for i in range(3):

        net = BayesNet(parse_bif_spec("resources/hailfinder.xml"))
        evidences, map_variables = create_random_evidence(
            len(net.nodes), net, 10, 50
        )
        # print(len(evidences), len(map_variables)), print(len(net.nodes))
        with benchmark("mpe") as mpe:
            ris_mpe = mpe_ask(evidences, net)

        with benchmark("map") as map:
            ris_map = map_ask(map_variables, evidences, net)

    print(
        "-------------------------------------------------------------------------------------"
    )

    print(
        "-------------------------------------------------------------------------------------"
    )
    print("win95pts - DOMINIO GRANDE 50% evidenze - 10% map")
    # for i in range(3):

    net = BayesNet(parse_bif_spec("resources/win95pts.xml"))
    evidences, map_variables = create_random_evidence(
        len(net.nodes), net, 10, 50
    )
    # print(len(evidences), len(map_variables)), print(len(net.nodes))
    with benchmark("mpe") as mpe:
        ris_mpe = mpe_ask(evidences, net)

    with benchmark("map") as map:
        ris_map = map_ask(map_variables, evidences, net)

    print(
        "-------------------------------------------------------------------------------------"
    )


if __name__ == "__main__":
    main()
