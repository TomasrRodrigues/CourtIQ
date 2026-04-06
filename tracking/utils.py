import math


def distance(c1, c2):
    """
    Computes Euclidean distance between two 2D points.

    Args:
        c1 (tuple[float, float] | list[float]): First point as (x, y).
        c2 (tuple[float, float] | list[float]): Second point as (x, y).

    Returns:
        float: Distance between c1 and c2 in pixels.
    """
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


def iou(b1, b2):
    """
    Computes Intersection over Union (IoU) between two axis-aligned boxes.

    Box format is [x1, y1, x2, y2], where (x1, y1) is top-left and
    (x2, y2) is bottom-right.

    Args:
        b1 (list[float] | tuple[float, float, float, float]): First box.
        b2 (list[float] | tuple[float, float, float, float]): Second box.

    Returns:
        float: IoU in [0, 1]. Returns 0 when boxes do not overlap.
    """
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

    # Small epsilon prevents division-by-zero for degenerate boxes.
    return inter / (area1 + area2 - inter + 1e-6)