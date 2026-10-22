"""Quantification of Epicardial Adipose Tissue (EAT) from a predicted mask."""
import numpy as np

from config import EFFECTIVE_SPACING_MM, SLICE_THICKNESS_MM


def eat_area_mm2(mask, spacing_mm=EFFECTIVE_SPACING_MM):
    """Area of one slice: number of EAT pixels x (pixel spacing)^2."""
    return float((np.asarray(mask) > 0).sum()) * spacing_mm * spacing_mm


def eat_area_cm2(mask, spacing_mm=EFFECTIVE_SPACING_MM):
    return eat_area_mm2(mask, spacing_mm) / 100.0


def eat_volume_cm3(masks, spacing_mm=EFFECTIVE_SPACING_MM, thickness_mm=SLICE_THICKNESS_MM):
    """Volume of a stack of slices: sum(slice areas) x slice thickness (mm3 -> cm3)."""
    total_mm2 = sum(eat_area_mm2(m, spacing_mm) for m in masks)
    return total_mm2 * thickness_mm / 1000.0
