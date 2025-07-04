
# Accuracy Reports

This directory contains comprehensive accuracy reports for the AI detection system.

## Report Files

- `latest_accuracy_report.json` - Always contains the most recent report
- `accuracy_report_YYYYMMDD_HHMMSS.json` - Timestamped reports for historical tracking

## Report Contents

Each report includes:
- **Summary**: Overall accuracy metrics
- **Performance Trends**: How accuracy changes over time  
- **Problem Patterns**: Specific issues identified in feedback
- **Recommendations**: Actionable improvements
- **Timeline**: Daily breakdown of feedback and accuracy
- **Model Insights**: Per-model performance analysis

## Automatic Generation

Reports are automatically generated when:
- 5 or more new feedback entries are received
- 24 hours have passed since the last report
- Manual generation via `/admin/generate-report` endpoint

## Accessing Reports

- **Latest Report**: `GET /admin/accuracy-report`
- **Generate New**: `POST /admin/generate-report`
- **Monitor Status**: `GET /admin/monitoring-status`
