from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
from scipy.ndimage import gaussian_filter

_FWHM_TO_SIGMA = 1.0 / 2.3548200450309493


@dataclass(frozen=True)
class VisitCombination:
    target_psf_fwhm_arcsec: float
    convolution_sigma_pixels: dict[str, float]
    weight_mode: str
    both_visit_fraction: float
    single_visit_fraction: float
    missing_fraction: float

    def to_dict(self) -> dict:
        return asdict(self)


def _image(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D image")
    return array


def _positive(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _normalized_gaussian(array: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 1e-8:
        return array.copy()
    finite = np.isfinite(array)
    numerator = gaussian_filter(np.where(finite, array, 0.0), sigma, mode="reflect")
    coverage = gaussian_filter(finite.astype(float), sigma, mode="reflect")
    output = np.full(array.shape, np.nan, dtype=float)
    supported = coverage > 1e-8
    output[supported] = numerator[supported] / coverage[supported]
    return output


def _weight(value: float | np.ndarray | None, shape: tuple[int, int], name: str) -> np.ndarray:
    if value is None:
        return np.ones(shape, dtype=float)
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        scalar = float(array)
        if not np.isfinite(scalar) or scalar < 0:
            raise ValueError(f"Weight {name} must be non-negative and finite")
        return np.full(shape, scalar, dtype=float)
    if array.shape != shape:
        raise ValueError(f"Weight {name} shape {array.shape} does not match image shape {shape}")
    return np.where(np.isfinite(array) & (array > 0), array, 0.0)


def combine_i_visits(
    images: Mapping[str, np.ndarray],
    psf_fwhm_arcsec: Mapping[str, float],
    *,
    pixscale_arcsec: float,
    inverse_variance: Mapping[str, float | np.ndarray] | None = None,
) -> tuple[np.ndarray, VisitCombination]:
    bands = ("i1", "i2")
    missing_images = [band for band in bands if band not in images]
    missing_psf = [band for band in bands if band not in psf_fwhm_arcsec]
    if missing_images:
        raise ValueError(f"Missing KiDS visit bands: {missing_images}")
    if missing_psf:
        raise ValueError(f"Missing visit PSF bands: {missing_psf}")
    arrays = {band: _image(band, images[band]) for band in bands}
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"KiDS visit images must share one shape, got {sorted(shapes)}")
    shape = next(iter(shapes))
    pixscale = _positive("pixscale_arcsec", pixscale_arcsec)
    psf = {band: _positive(f"PSF FWHM {band}", psf_fwhm_arcsec[band]) for band in bands}
    target = max(psf.values())
    target_sigma = target * _FWHM_TO_SIGMA / pixscale
    matched: dict[str, np.ndarray] = {}
    sigmas: dict[str, float] = {}
    for band in bands:
        input_sigma = psf[band] * _FWHM_TO_SIGMA / pixscale
        sigma = float(np.sqrt(max(0.0, target_sigma**2 - input_sigma**2)))
        sigmas[band] = sigma
        matched[band] = _normalized_gaussian(arrays[band], sigma)

    weights = {
        band: _weight(
            None if inverse_variance is None else inverse_variance.get(band),
            shape,
            band,
        )
        for band in bands
    }
    valid = {band: np.isfinite(matched[band]) & (weights[band] > 0) for band in bands}
    effective = {band: np.where(valid[band], weights[band], 0.0) for band in bands}
    denominator = effective["i1"] + effective["i2"]
    numerator = (
        np.where(valid["i1"], matched["i1"], 0.0) * effective["i1"]
        + np.where(valid["i2"], matched["i2"], 0.0) * effective["i2"]
    )
    combined = np.full(shape, np.nan, dtype=float)
    supported = denominator > 0
    combined[supported] = numerator[supported] / denominator[supported]
    support_count = valid["i1"].astype(np.int8) + valid["i2"].astype(np.int8)
    total = float(combined.size)
    return combined, VisitCombination(
        target_psf_fwhm_arcsec=target,
        convolution_sigma_pixels=sigmas,
        weight_mode="inverse_variance" if inverse_variance is not None else "equal_finite_visit",
        both_visit_fraction=float(np.sum(support_count == 2) / total),
        single_visit_fraction=float(np.sum(support_count == 1) / total),
        missing_fraction=float(np.sum(support_count == 0) / total),
    )


def prepare_ugri_from_kids_visits(
    images: Mapping[str, np.ndarray],
    psf_fwhm_arcsec: Mapping[str, float],
    *,
    pixscale_arcsec: float,
    inverse_variance: Mapping[str, float | np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float], VisitCombination]:
    required = ("u", "g", "r", "i1", "i2")
    missing_images = [band for band in required if band not in images]
    missing_psf = [band for band in required if band not in psf_fwhm_arcsec]
    if missing_images:
        raise ValueError(f"Missing KiDS science bands: {missing_images}")
    if missing_psf:
        raise ValueError(f"Missing KiDS PSF bands: {missing_psf}")
    base = {band: _image(band, images[band]) for band in ("u", "g", "r")}
    i_image, metadata = combine_i_visits(
        images,
        psf_fwhm_arcsec,
        pixscale_arcsec=pixscale_arcsec,
        inverse_variance=inverse_variance,
    )
    shapes = {array.shape for array in (*base.values(), i_image)}
    if len(shapes) != 1:
        raise ValueError(f"Prepared bands must share one shape, got {sorted(shapes)}")
    return (
        {**base, "i": i_image},
        {
            "u": _positive("PSF FWHM u", psf_fwhm_arcsec["u"]),
            "g": _positive("PSF FWHM g", psf_fwhm_arcsec["g"]),
            "r": _positive("PSF FWHM r", psf_fwhm_arcsec["r"]),
            "i": metadata.target_psf_fwhm_arcsec,
        },
        metadata,
    )


def psf_match_bands(
    images: Mapping[str, np.ndarray],
    psf_fwhm_arcsec: Mapping[str, float],
    *,
    pixscale_arcsec: float,
) -> tuple[dict[str, np.ndarray], float, dict[str, float]]:
    if not images:
        raise ValueError("At least one band is required")
    names = tuple(images)
    arrays = {band: _image(band, images[band]) for band in names}
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"Bands must share one shape, got {sorted(shapes)}")
    missing = [band for band in names if band not in psf_fwhm_arcsec]
    if missing:
        raise ValueError(f"Missing PSF bands: {missing}")
    pixscale = _positive("pixscale_arcsec", pixscale_arcsec)
    psf = {band: _positive(f"PSF FWHM {band}", psf_fwhm_arcsec[band]) for band in names}
    target = max(psf.values())
    target_sigma = target * _FWHM_TO_SIGMA / pixscale
    output: dict[str, np.ndarray] = {}
    sigmas: dict[str, float] = {}
    for band in names:
        input_sigma = psf[band] * _FWHM_TO_SIGMA / pixscale
        sigma = float(np.sqrt(max(0.0, target_sigma**2 - input_sigma**2)))
        sigmas[band] = sigma
        output[band] = _normalized_gaussian(arrays[band], sigma)
    return output, target, sigmas


def robust_background_normalize(
    image: np.ndarray,
    *,
    border_fraction: float = 0.125,
) -> tuple[np.ndarray, dict[str, float]]:
    array = _image("image", image)
    if not 0 < border_fraction < 0.5:
        raise ValueError("border_fraction must lie in (0, 0.5)")
    finite = np.isfinite(array)
    if not finite.any():
        raise ValueError("Image contains no finite pixels")
    height, width = array.shape
    border = max(4, int(round(min(height, width) * border_fraction)))
    mask = np.zeros(array.shape, dtype=bool)
    mask[:border, :] = True
    mask[-border:, :] = True
    mask[:, :border] = True
    mask[:, -border:] = True
    sample = array[finite & mask]
    if sample.size < 32:
        sample = array[finite]
    location = float(np.median(sample))
    mad = float(np.median(np.abs(sample - location)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(sample))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Unable to estimate a positive background scale")
    output = np.full(array.shape, np.nan, dtype=np.float32)
    output[finite] = np.clip((array[finite] - location) / scale, -8.0, 30.0)
    return output, {"background_median": location, "background_sigma_mad": float(scale)}
