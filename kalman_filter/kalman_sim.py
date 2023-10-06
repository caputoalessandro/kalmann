from itertools import islice

import matplotlib.pyplot as pl
from matplotlib import rcParams

from kalman_filter.kalman import simulate_moving_object

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Roboto"]


def plot_variable(real_states, measurements, kalman_predictions):
    time = list(range(len(real_states)))
    pl.grid(True)
    pl.plot(time, real_states, label="Stato reale dell'oggetto", alpha=0.8)
    pl.plot(
        time,
        kalman_predictions,
        label="Stato predetto dal KF",
        alpha=0.7,
        c="red",
    )
    pl.scatter(time, measurements, label="Misurazioni", marker="x", alpha=0.8)


def plot_results(real_states, measurements, kalman_predictions, title):
    pl.figure(figsize=(6, 5.5))
    if title:
        pl.suptitle(title)
    x_data = (
        [state[0][0] for state in real_states],
        [measurement[0][0] for measurement in measurements],
        [kr.state[0][0] for kr in kalman_predictions],
    )
    pl.subplot(2, 1, 1)
    pl.ylabel("posizione")
    plot_variable(*x_data)
    pl.legend(loc="upper left", fancybox=True, framealpha=0.7)
    y_data = (
        [state[1][0] for state in real_states],
        [measurement[1][0] for measurement in measurements],
        [kr.state[1][0] for kr in kalman_predictions],
    )
    pl.subplot(2, 1, 2)
    pl.xlabel("istante (t)")
    pl.ylabel("velocità")
    plot_variable(*y_data)
    pl.tight_layout()
    pl.show()


def kalman_simulation(iterations=30, title=None, *args, **kwargs):
    results = list(islice(simulate_moving_object(*args, **kwargs), iterations))
    real_states = [r[0] for r in results]
    measurements = [r[1] for r in results]
    kalman_results = [r[2] for r in results]
    plot_results(real_states, measurements, kalman_results, title)

    return results
