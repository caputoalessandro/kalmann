from aima_probability import *


# def create_random_evidence(N: int, net: BayesNet, percentage):
#
#     keys = random.sample(net.variables, N)
#     real_percentage = int((N / 100) * percentage)
#     evidences_keys = keys[:real_percentage]
#     map_keys = keys[real_percentage:]
#     evidences = {}
#     for key in evidences_keys:
#         domain = net.variable_values(key)
#         evidences[key] = random.choice(domain)
#
#     return evidences, map_keys


def create_random_evidence(N: int, net: BayesNet, evidence_perc, map_perc):

    keys = random.sample(net.variables, N)
    evidence_real_percentage = int((N / 100) * evidence_perc)
    map_real_percentage = int((N / 100) * map_perc)
    # total_perc = evidence_perc + map_perc
    # limit = 100 - total_perc
    limit = evidence_real_percentage + map_real_percentage
    evidences_keys = keys[:evidence_real_percentage]
    map_keys = keys[evidence_real_percentage:limit]
    evidences = {}
    for key in evidences_keys:
        domain = net.variable_values(key)
        evidences[key] = random.choice(domain)

    return evidences, map_keys
