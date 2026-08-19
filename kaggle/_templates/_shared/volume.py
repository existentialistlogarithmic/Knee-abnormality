"""DICOM series to one laterality-corrected, mm-resampled volume.

Included verbatim into generated kernels by eda/generate_kernels.py.
Kaggle script kernels are single files, so sharing code means splicing it
at generation time. Editing it here changes every kernel that includes it.
"""

def normalise(volume: np.ndarray) -> np.ndarray:
    """Percentile clip then scale to uint8.

    Per-volume rather than per-slice: MRI intensity is arbitrary between studies
    but consistent within one acquisition, and per-slice normalisation would
    destroy the relative brightness that distinguishes fluid from fat.
    """
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return np.zeros_like(volume, dtype=np.uint8)
    low, high = np.percentile(finite, [1.0, 99.0])
    if high <= low:
        high = low + 1.0
    # Non-finite values must be pinned before the cast: NaN -> uint8 is
    # undefined and would write arbitrary bytes into the cache silently.
    filled = np.nan_to_num(volume, nan=low, posinf=high, neginf=low)
    scaled = (np.clip(filled, low, high) - low) / (high - low)
    return (scaled * 255.0).astype(np.uint8)


def resize(image: np.ndarray, size: int) -> np.ndarray:
    try:
        import cv2

        return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    except ImportError:
        from PIL import Image

        return np.asarray(Image.fromarray(image).resize((size, size), Image.BILINEAR))


def pick_slices(count: int, wanted: int) -> list[int]:
    """Evenly spaced through the stack, always including both ends.

    Centre-cropping would be wrong here: meniscal tears sit at the periphery of
    the sagittal stack, exactly where a centre crop throws data away.
    """
    if count <= 0:
        return []
    if count <= wanted:
        return list(range(count)) + [count - 1] * (wanted - count)
    return list(np.linspace(0, count - 1, wanted).round().astype(int))


def read_series_volume(directory: Path) -> tuple[np.ndarray | None, float | None, str | None]:
    """Return (volume, mm_per_pixel, laterality) with slices in anatomical order."""
    try:
        names = sorted(e.name for e in os.scandir(directory) if e.name.endswith(".dcm"))
    except FileNotFoundError:
        return None, None, None
    if not names:
        return None, None, None

    slices = []
    spacing = None
    laterality = None
    for name in names:
        try:
            ds = pydicom.dcmread(str(directory / name), force=True)
            pixels = ds.pixel_array.astype(np.float32)
        except Exception:  # noqa: BLE001 - one unreadable slice must not lose the series
            continue
        if pixels.ndim != 2:
            continue
        position = getattr(ds, "ImagePositionPatient", None)
        order = float(position[2]) if position is not None and len(position) == 3 else len(slices)
        if spacing is None:
            value = getattr(ds, "PixelSpacing", None)
            if value is not None and len(value) >= 1:
                spacing = float(value[0])
        if laterality is None:
            laterality = (getattr(ds, "Laterality", None)
                          or _laterality_from_description(getattr(ds, "SeriesDescription", "")))
        slices.append((order, pixels))

    if not slices:
        return None, None, None
    slices.sort(key=lambda item: item[0])
    return np.stack([s[1] for s in slices]), spacing, laterality


def _laterality_from_description(description: str) -> str | None:
    text = (description or "").upper()
    if text.startswith("LT") or "_LT_" in text or " LEFT" in text or text.startswith("L_"):
        return "L"
    if text.startswith("RT") or "_RT_" in text or " RIGHT" in text or text.startswith("R_"):
        return "R"
    return None


def build_study(root: Path, split: str, study: str, series_rows: pd.DataFrame) -> tuple:
    """One study to (planes, slices, size, size) uint8, plus a record of what happened."""
    record = {"StudyInstanceUID": study, "split": split, "laterality": None,
              "mirrored": False, "planes_found": 0, "missing_planes": [], "error": None}
    channels = []

    for plane in PLANES:
        candidates = series_rows[(series_rows.Anatomical_Plane == plane)
                                 & (series_rows.Fluid_Sensitive == 1)]
        if candidates.empty:
            candidates = series_rows[series_rows.Anatomical_Plane == plane]
        if candidates.empty:
            record["missing_planes"].append(plane)
            channels.append(np.zeros((SLICES_PER_PLANE, TARGET_SIZE, TARGET_SIZE), np.uint8))
            continue

        # Prefer the series with the most slices — the diagnostic acquisition
        # rather than a localiser.
        chosen = candidates.sort_values("n_slices", ascending=False).iloc[0]
        directory = root / f"{split}_series" / study / chosen.SeriesInstanceUID
        volume, spacing, laterality = read_series_volume(directory)
        if volume is None:
            record["missing_planes"].append(plane)
            channels.append(np.zeros((SLICES_PER_PLANE, TARGET_SIZE, TARGET_SIZE), np.uint8))
            continue

        record["laterality"] = record["laterality"] or laterality
        record["planes_found"] += 1

        indices = pick_slices(len(volume), SLICES_PER_PLANE)
        volume = volume[indices]

        # Physical resampling: crop or pad to the field of view we want, then
        # resize once. Doing it in this order keeps millimetres meaningful.
        if spacing and spacing > 0:
            wanted_pixels = int(round(TARGET_SIZE * TARGET_MM_PER_PIXEL / spacing))
            wanted_pixels = max(8, min(wanted_pixels, max(volume.shape[1], volume.shape[2])))
            centre_y, centre_x = volume.shape[1] // 2, volume.shape[2] // 2
            half = wanted_pixels // 2
            y0, y1 = max(0, centre_y - half), min(volume.shape[1], centre_y + half)
            x0, x1 = max(0, centre_x - half), min(volume.shape[2], centre_x + half)
            volume = volume[:, y0:y1, x0:x1]

        volume = normalise(volume)
        resized = np.stack([resize(frame, TARGET_SIZE) for frame in volume])
        channels.append(resized)

    stack = np.stack(channels)  # (planes, slices, size, size)

    if (record["laterality"] or "").upper().startswith("R"):
        stack = stack[..., ::-1].copy()
        record["mirrored"] = True

    return stack, record
