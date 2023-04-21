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

log_dir_mmoex = '/itf-fi-ml/home/sebasam/LFDeep/outputs/2023-04-11/14-19-30/events.out.tfevents.1681215592.ml6.hpc.uio.no.704782.0'
log_dir_MH = '/itf-fi-ml/home/sebasam/LFDeep/outputs/2023-04-18/19-39-36/events.out.tfevents.1681839594.ml6.hpc.uio.no.575898.0'

# Generate the list of element error metrics
tags = [f"metric/val/element_errors_{i}" for i in range(1, 640)]

metrics_df_mmoex = extract_metrics(log_dir_mmoex, tags)
metrics_df_MH = extract_metrics(log_dir_MH, tags)
#print(metrics_df)

# Save the extracted metrics to a CSV file
metrics_df_mmoex.to_csv('/itf-fi-ml/home/sebasam/LFDeep/element_errors/mmoeex_element_errors_metrics.csv')
metrics_df_MH.to_csv('/itf-fi-ml/home/sebasam/LFDeep/element_errors/MH_element_errors_metrics.csv')
