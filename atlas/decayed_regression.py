"""Experimental date-balanced, exponentially weighted ridge for paper research."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from math import fsum, hypot, isfinite, sqrt

from .regression import MIN_TRAIN_DATES, Label, Regression, _solve


def _date_weights(labels: Sequence[Label], half_life_dates: float):
    if isinstance(half_life_dates, bool) or not isfinite(half_life_dates) or half_life_dates <= 0:
        raise ValueError("half-life must be a positive finite number of training dates")
    dates = sorted({label.observation.session_date for label in labels})
    if len(dates) < MIN_TRAIN_DATES:
        raise ValueError(f"need at least {MIN_TRAIN_DATES} matured training dates")
    weights = {day: 2.0**(-(len(dates)-index-1)/half_life_dates) for index,day in enumerate(dates)}
    total = fsum(weights.values())
    effective = total*total/fsum(value*value for value in weights.values())
    return weights, total, effective


def effective_dates(labels: Sequence[Label], half_life_dates: float) -> float:
    return _date_weights(labels, half_life_dates)[2]


def fit_decay(labels: Sequence[Label], alpha: float, half_life_dates: float) -> Regression:
    """Center/scale within the training set and weight dates independently of stock count."""
    if not isfinite(alpha) or alpha < 0:
        raise ValueError("regularization must be finite and nonnegative")
    date_weights, total, effective = _date_weights(labels, half_life_dates)
    if effective+1e-12 < MIN_TRAIN_DATES:
        raise ValueError(f"need at least {MIN_TRAIN_DATES} weight-effective training dates")
    samples = [label.observation.values for label in labels]
    if not samples[0] or any(len(row) != len(samples[0]) for row in samples):
        raise ValueError("training features must have consistent nonempty dimensions")
    if any(not isfinite(label.forward_return) or not all(isfinite(value) for value in label.observation.values)
           for label in labels):
        raise ValueError("training features and targets must be finite")
    counts = Counter(label.observation.session_date for label in labels)
    weights = [date_weights[label.observation.session_date]/total/counts[label.observation.session_date]
               for label in labels]
    try:
        means = tuple(fsum(row[column]*weight for row,weight in zip(samples,weights)) for column in range(len(samples[0])))
        raw_scales = tuple(hypot(*((row[column]-mean)*sqrt(weight) for row,weight in zip(samples,weights) if weight))
                           for column,mean in enumerate(means))
        scales = tuple(value or 1.0 for value in raw_scales)
        intercept = fsum(label.forward_return*weight for label,weight in zip(labels,weights))
        if not all(isfinite(value) for value in (*means,*scales,intercept)):
            raise ValueError("weighted fit exceeds the finite numerical range")
        active = [index for index,value in enumerate(raw_scales) if value > 0]
        standardized = [[(row[index]-means[index])/scales[index] for row in samples] for index in active]
        centered = [label.forward_return-intercept for label in labels]
        gram = [[fsum(a*b*w for a,b,w in zip(left,right,weights))+(alpha if i == j else 0)
                 for j,right in enumerate(standardized)] for i,left in enumerate(standardized)]
        target = [fsum(value*y*w for value,y,w in zip(column,centered,weights)) for column in standardized]
        solved = dict(zip(active,_solve(gram,target),strict=True))
        return Regression(means,scales,tuple(solved.get(index,0.0) for index in range(len(means))),intercept)
    except OverflowError as error:
        raise ValueError("weighted fit exceeds the finite numerical range") from error
