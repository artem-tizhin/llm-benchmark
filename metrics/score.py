def safe_mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0