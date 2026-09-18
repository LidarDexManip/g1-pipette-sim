"""Geometric seating gate; distances are meters in the tip's local frame."""
import numpy as np


def attachment_ready(error, angle_deg, settings):
    """Negative axial error means the nozzle has entered the tip mouth.

    Explicit insertion limits replace the legacy symmetric axial tolerance.
    The caller requires this condition continuously for dwell_seconds.
    """
    if 'min_insertion_depth' in settings:
        lower = settings['min_insertion_depth']
        upper = settings['max_insertion_depth']
        if not 0 < lower <= upper:
            raise ValueError('Require 0 < min_insertion_depth <= max_insertion_depth')
        axial_ok = lower <= -error[2] <= upper
    else:
        axial_ok = abs(error[2]) < settings['axial_tolerance']
    return bool(axial_ok and np.linalg.norm(error[:2]) < settings['radial_tolerance']
                and angle_deg < settings['angle_tolerance_deg'])
