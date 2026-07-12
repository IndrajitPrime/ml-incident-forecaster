# ML Incident Volume Forecaster

A desktop application that uses Random Forest machine learning to predict future IT support ticket volumes by assignment group, priority, and category. Built with Python, scikit-learn, and Tkinter.

## The Problem It Solves

Support teams managing IT incident queues frequently face unpredictable volume spikes that overwhelm available capacity and breach SLA targets. Traditional approaches to workload planning rely on gut feel or simple averages. This tool learns from historical ticket data to forecast future volumes and flag spike days before they happen, giving managers time to reallocate resources proactively.

## How It Works

The application trains separate Random Forest Regressor models for each dimension of the ticket data:

- Assignment Group models: predict monthly ticket volumes per team
- Priority models: predict volume distribution across priority levels
- Category / Sub-Category models: predict which issue types will dominate
- Spike day detection: identifies specific future dates likely to exceed normal thresholds based on learned daily patterns

The models use time-series features extracted from historical ticket dates (day of week, month, rolling averages) combined with categorical features (assignment group, priority, category) encoded via Label Encoding. Trained models are persisted to disk using pickle so predictions can be regenerated without retraining.

## Validation

Each model is evaluated on a chronological 80/20 train/test split (not random) to avoid data leakage common in time-series evaluation, reporting R-squared and mean absolute error (MAE) per model, printed to console on each training run. Performance varies by data volume and signal strength: higher-volume assignment-group and priority-level models show moderate predictive power (R-squared roughly 0.2 to 0.5 in testing), while sparse, granular category/sub-category models are weaker, some near-zero or negative R-squared, reflecting the difficulty of forecasting low-frequency events.

Testing was conducted using synthetic ticket data structured to resemble realistic IT support patterns, not live production data.

## Input

An Excel file (.xlsx or .xls) with the following required columns:

- Assignment Group: Team or queue the ticket was routed to
- Priority: Ticket priority level
- Created: Date the ticket was created
- Category: Ticket category
- Sub Category: Ticket sub-category

## Output

The application generates four visualization tabs:

- Assignment Group Volumes: bar chart of predicted monthly volumes per team
- Priority Volumes: pie chart of predicted priority distribution
- Category Volumes: horizontal bar chart of top predicted issue categories
- Spike Days: timeline of predicted high-volume days with assignment group breakdown

## Files

- SupportPredict.py: Base version of the application
- SupportValue.py: Enhanced version with improved visualizations, scrollable frames, and color-coded charts

## Tech Stack

- Python 3.x
- scikit-learn (Random Forest Regressor, Label Encoder)
- Pandas / NumPy
- Matplotlib
- Tkinter (desktop UI)
- pickle (model persistence)

## How To Run

Install dependencies:

pip install scikit-learn pandas numpy matplotlib python-dateutil openpyxl

Run the enhanced version:

python SupportValue.py

Browse to your historical ticket data Excel file, set the number of months to predict, and click Process & Predict.

## Background

This tool was built independently to address a real operational gap in incident management planning. It was developed and validated using hypothetical/synthetic support ticket data designed to reflect realistic volume and priority patterns.

## License

Apache License 2.0
