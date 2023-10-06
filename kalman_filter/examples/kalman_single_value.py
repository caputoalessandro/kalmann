from typing import NamedTuple


class KalmanResult(NamedTuple):
    estimate: float
    estimate_err: float
    gain: float


def kalman_filter(estimate, estimate_err, measurements, measurement_err):
    for measurement in measurements:
        gain = get_kalman_gain(estimate_err, measurement_err)
        estimate = get_current_estimate(estimate, measurement, gain)
        estimate_err = get_estimate_err(gain, estimate_err)
        yield KalmanResult(estimate, estimate_err, gain)


def get_kalman_gain(estimate_error, measurement_error):
    """
    Restituisce il Kalman Gain, ovvero il rapporto dell'accuratezza della stima
    rispetto all'accuratezza della misura.
    """
    return estimate_error / (estimate_error + measurement_error)


def get_current_estimate(previous_estimate, measurement, gain):
    return previous_estimate + gain * (measurement - previous_estimate)


def get_estimate_err(gain, estimate_error):
    return (1 - gain) * estimate_error


if __name__ == "__main__":
    from pprint import pprint
    import pandas as pd
    import altair as alt

    # Valore vero: 72
    # Esempio da https://www.youtube.com/watch?v=SIQJaqYVtuE
    results = list(kalman_filter(68, 2, [75, 71, 70, 74], 4))
    results_df = pd.DataFrame(results)
