# -*- coding: utf-8 -*-

import numpy as np
import random
from scipy.special import comb

def _get_polynomial_array(nTimes=100000, nPoints=4):
    def bernstein_poly(i, n, t):
        return comb(n, i) * (t ** (n - i)) * (1 - t) ** i

    t = np.linspace(0.0, 1.0, nTimes)
    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)]).astype(np.float32)
    return polynomial_array

def get_bezier_curve(points):
    polynomial_array = _get_polynomial_array()
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)
    return xvals, yvals

def non_linear_transformation(inputs, inverse=False, inverse_prop=0.5, nPoints=4):
    # inputs = np.array(inputs)
    start_point, end_point = inputs.min(), inputs.max()
    xPoints = [start_point, end_point]
    yPoints = [start_point, end_point]
    for _ in range(nPoints - 2):
        xPoints.insert(1, random.uniform(xPoints[0], xPoints[-1]))
        yPoints.insert(1, random.uniform(yPoints[0], yPoints[-1]))
    xvals, yvals = get_bezier_curve([[x, y] for x, y in zip(xPoints, yPoints)])
    if inverse and random.random() <= inverse_prop:
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    return np.interp(inputs, xvals, yvals)


def location_scale_transformation(inputs, slide_limit=10, vrange=(0., 1.)): # 20
    scale = np.array(max(min(random.gauss(1, 0.1), 1.1), 0.9), dtype=np.float32)
    location = np.array(random.gauss(0, 0.5), dtype=np.float32)
    location = np.clip(location, vrange[0] - np.percentile(inputs, slide_limit),
                       vrange[1] - np.percentile(inputs, 100 - slide_limit))
    return np.clip(inputs * scale + location, vrange[0], vrange[1])


def Global_Location_Scale_Augmentation(image):
    image = non_linear_transformation(image, inverse=False)
    image = location_scale_transformation(image).astype(np.float32)
    return image


def Local_Location_Scale_Augmentation(image, mask, background_threshold = 0.01, vrange=(0.,1.)):
    output_image = np.zeros_like(image)

    mask = mask.astype(np.int32)

    output_image[mask == 0] = location_scale_transformation(
        non_linear_transformation(image[mask == 0], inverse=True, inverse_prop=1))

    for c in range(1, np.max(mask) + 1):
        if (mask == c).sum() == 0: continue
        output_image[mask == c] = location_scale_transformation(
            non_linear_transformation(image[mask == c], inverse=True, inverse_prop=0.5))

    if background_threshold >= vrange[0]:
        output_image[image <= background_threshold] = image[image <= background_threshold]

    return output_image
