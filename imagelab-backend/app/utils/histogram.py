import cv2
import numpy as np


def compute_histogram(image: np.ndarray) -> dict:
    """Compute per-channel histogram and statistics for an image.

    Returns a dict with a ``channels`` key containing a list of per-channel
    dicts.  Each channel dict has: ``name``, ``mean``, ``std``, ``min``,
    ``max``, and ``histogram`` (256 bins as ints).

    Grayscale images (ndim == 2) produce a single "Gray" channel.
    Colour images produce "Blue", "Green", "Red" channels (BGR order).
    """
    if image.ndim == 2:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten().astype(int).tolist()
        return {
            "channels": [
                {
                    "name": "Gray",
                    "mean": float(np.mean(image)),
                    "std": float(np.std(image)),
                    "min": int(np.min(image)),
                    "max": int(np.max(image)),
                    "histogram": hist,
                }
            ]
        }

    channel_names = ["Blue", "Green", "Red"]
    channels = []
    for i, name in enumerate(channel_names[: image.shape[2]]):
        ch = image[:, :, i]
        hist = cv2.calcHist([image], [i], None, [256], [0, 256]).flatten().astype(int).tolist()
        channels.append(
            {
                "name": name,
                "mean": float(np.mean(ch)),
                "std": float(np.std(ch)),
                "min": int(np.min(ch)),
                "max": int(np.max(ch)),
                "histogram": hist,
            }
        )
    return {"channels": channels}
