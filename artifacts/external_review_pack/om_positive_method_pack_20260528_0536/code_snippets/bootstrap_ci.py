"""Small percentile bootstrap median CI."""
import random
from statistics import median

def bootstrap_median_ci(values, samples=1000, seed=20260508):
    if not values:
        return {"ci95_low": None, "ci95_high": None}
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        draws.append(median(rng.choice(values) for _ in values))
    draws.sort()
    return {"ci95_low": draws[int(0.025 * samples)], "ci95_high": draws[int(0.975 * samples) - 1]}
