"""Small metric helper required by the original Bailando VQ-VAE."""


def average_metrics(metric_rows):
    values = {}
    for row in metric_rows:
        for key, value in row.items():
            values.setdefault(key, []).append(value)
    return {
        key: sum(items) / len(items)
        for key, items in values.items()
        if items
    }
