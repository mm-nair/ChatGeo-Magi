import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT / 'revision_workspace/api_investigation/live/london_1999_2026.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'revision_workspace/api_investigation/london_new_demonstration')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT / 'revision_workspace'):
        parser.error('Output must remain in revision_workspace')
    payload = json.loads(args.input.read_text())['payload']
    dates = np.array([row['date'] for row in payload['result']], dtype=float)
    values = np.array([row['totalintensity'] for row in payload['result']], dtype=float)
    if not np.isfinite(values).all() or not np.all(np.diff(dates) > 0):
        raise ValueError('Invalid London series')
    endpoint_rate = (values[-1] - values[0]) / (dates[-1] - dates[0])
    slope = np.polyfit(dates - dates[0], values, 1)[0]
    summary = {'label': 'New live NOAA demonstration, not reconstructed historical output', 'returned_model': payload.get('model'), 'calculator_version': payload.get('version'), 'first_year': dates[0], 'last_year': dates[-1], 'first_intensity_nT': values[0], 'last_intensity_nT': values[-1], 'net_change_nT': values[-1] - values[0], 'endpoint_nT_per_year': endpoint_rate, 'endpoint_nT_per_decade': endpoint_rate * 10, 'ols_nT_per_year': slope, 'points': len(dates), 'latitude': payload['result'][0]['latitude'], 'longitude': payload['result'][0]['longitude'], 'elevation_km': payload['result'][0]['elevation'], 'date_note': 'Annual January 1 model estimates, including 2026; not completed-calendar-year observations'}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    with (args.output / 'series.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['decimal_year', 'total_intensity_nT', 'returned_model'])
        writer.writerows(zip(dates, values, [payload.get('model')] * len(dates)))
    figure, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True, layout='constrained')
    axes[0].plot(dates, values, marker='o', markersize=3)
    axes[0].set(ylabel='Total field intensity (nT)', ylim=(0, 60000), title=f"New NOAA demonstration: {payload.get('model')}, 51.5°N, 0.12°W")
    axes[1].plot(dates, values - values[0], marker='o', markersize=3)
    axes[1].set(ylabel='Change from 1999 (nT)', xlabel='Year (January 1)', ylim=(0, max(1000, 1.1 * (values.max() - values[0]))))
    for extension in ('pdf', 'png'):
        figure.savefig(args.output / f'london_new.{extension}', dpi=400)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
