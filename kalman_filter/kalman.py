from typing import NamedTuple
import numpy as np
from numpy.linalg import inv


class KalmanResult(NamedTuple):
    state: np.ndarray
    state_cov: np.ndarray
    gain: np.ndarray


def calculate_movement(state: np.ndarray, timedelta: float, control: float):
    A = np.array([[1, timedelta], [0, 1]])
    B = np.array([[0.5 * timedelta ** 2], [timedelta]])
    return A @ state + B * control


def get_a_priori_error_cov(state_cov, timedelta, Q=np.zeros((2, 2))):
    """
    Q: covarianza dovuta al rumore del processo
    """
    A = np.array([[1, timedelta], [0, 1]])
    return A @ state_cov @ A.T + Q


def get_a_posteriori_estimate(a_priori_estimate, measurement, gain):
    return a_priori_estimate + gain @ (measurement - a_priori_estimate)


def get_a_posteriori_error_cov(previous_cov, gain):
    return (np.identity(2) - gain) @ previous_cov


def get_gain(state_cov, mea_cov):
    """
    Calcola il Kalman Gain. La formula completa è P_kp H.T / (H P_kp H.T + R).
    H serve a convertire la forma della matrice in maniera compatibile a
    quella del Kalman Gain. In questo caso non c'è necessità di usarla
    perché sarebbe uguale all'identità.

    Il denominatore è uguale a S_k su wikipedia
    """
    return state_cov @ inv(state_cov + mea_cov)


def kalman_filter(
    state,
    state_cov,
    mea_cov,
    control=0,
    timedelta=1,
    pnoise_cov=np.zeros((2, 2)),
):
    measurement = yield
    while True:
        state = calculate_movement(state, timedelta, control)
        state_cov = get_a_priori_error_cov(state_cov, timedelta, pnoise_cov)
        gain = get_gain(state_cov, mea_cov)
        state = get_a_posteriori_estimate(state, measurement, gain)
        state_cov = get_a_posteriori_error_cov(state_cov, gain)
        measurement = yield KalmanResult(state, state_cov, gain)


def noise(covariance):
    sample = np.random.multivariate_normal(np.zeros(2), covariance)
    return sample.reshape(2, 1)


class SimResult(NamedTuple):
    state: np.ndarray
    measurement: np.ndarray
    kalman: KalmanResult


def simulate_moving_object(
    real_state,
    real_process_cov,
    real_sensor_cov,
    acceleration,
    kalman_state=None,
    kalman_state_cov=None,
    kalman_process_cov=None,
    kalman_sensor_cov=None,
    timedelta=1,
):
    kalman_state = kalman_state if kalman_state is not None else real_state
    kalman_state_cov = (
        kalman_state_cov if kalman_state_cov is not None else real_process_cov
    )
    kalman_process_cov = (
        kalman_process_cov
        if kalman_process_cov is not None
        else real_process_cov
    )
    kalman_sensor_cov = (
        kalman_sensor_cov if kalman_sensor_cov is not None else real_sensor_cov
    )

    kf = kalman_filter(
        kalman_state,
        kalman_state_cov,
        kalman_sensor_cov,
        control=acceleration,
        timedelta=timedelta,
        pnoise_cov=kalman_process_cov,
    )
    next(kf)
    while True:
        real_state = calculate_movement(real_state, timedelta, acceleration)
        real_state += noise(real_process_cov)
        measurement = real_state + noise(real_sensor_cov)
        kalman_prediction = kf.send(measurement)
        yield SimResult(real_state, measurement, kalman_prediction)
