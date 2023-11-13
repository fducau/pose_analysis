import pytest

from process import *


def test_estimate_abduction():
    # Make a 90 degrees abduction example
    pose_coordinates = [
        (0, 0),     # left hip
        (-1, 0),    # right hip
        (0, 1),    # left knee
        (-1, 1)    # right knee
    ]
    abd = estimate_abduction(pose_coordinates, limp_leg='LEFT')
    assert abd == 0

    # Make a 45 degrees abduction example
    pose_coordinates = [
        (0, 0),     # left hip
        (-1, 0),    # right hip
        (1, 1),    # left knee
        (-1, 1)    # right knee
    ]
    abd = estimate_abduction(pose_coordinates, limp_leg='LEFT')
    assert abd == 45

def test_estimate_flexion():
    # Make a 90 degrees abduction example
    pose_coordinates = [
        (0, 0),     # left hip
        (-1, 0),    # right hip
        (0, 1),    # left knee
        (-1, 1)    # right knee
    ]
    flx = estimate_flexion(pose_coordinates, limp_leg='LEFT')
    assert flx == 0

    # Make a 45 degrees abduction example
    pose_coordinates = [
        (0, 0),     # left hip
        (-1, 0),    # right hip
        (-1, 1),    # left knee
        (-1, 1)    # right knee
    ]
    flx = estimate_flexion(pose_coordinates, limp_leg='LEFT')
    assert flx == 45
