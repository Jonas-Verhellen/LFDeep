import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_metrics(log_dir, tags):
    ea = EventAccumulator(log_dir)
    ea.Reload()

    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        data[tag] = pd.DataFrame(events, columns=["wall_time", "step", tag]).drop("wall_time", axis=1).set_index("step")

    return pd.concat(data.values(), axis=1)

log_dir = '/itf-fi-ml/home/sebasam/LFDeep/outputs/2023-04-11/14-19-30/events.out.tfevents.1681215592.ml6.hpc.uio.no.704782.0'

# Generate the list of element error metrics
tags = [f"metric/val/element_errors_{i}" for i in range(1, 640)]

metrics_df = extract_metrics(log_dir, tags)
print(metrics_df)

# Save the extracted metrics to a CSV file
metrics_df.to_csv('/itf-fi-ml/home/sebasam/LFDeep/element_errors/mmoeex_element_errors_metrics.csv')
