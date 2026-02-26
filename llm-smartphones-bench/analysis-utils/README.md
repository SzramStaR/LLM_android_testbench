# Analysis Utils README

This directory contains Python scripts for aggregating, downloading, and visualizing performance / accuracy data from LLM benchmarks.

## Scripts

- **data_aggregator.py**: Reads and processes performance and evaluation data from JSON files, exports the aggregated results to an Excel file with multiple worksheets.
- **r2_downloader.py**: Downloads objects from a specified Cloudflare R2 bucket to a local directory.
- **plots_maker.py**: Loads processed evaluation and performance data to generate various plots including accuracy metrics, tokens per second, RAM usage, power draw, and temperature heatmaps, saving them as SVG files.
- **wandb_downloader.py**: Retrieves benchmark results from a Weights & Biases project for LM Evaluation Harness runs and saves them as individual JSON files in a local directory.