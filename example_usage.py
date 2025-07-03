
from database import get_analytics_summary
import json

# Get analytics data
analytics = get_analytics_summary()

# Print summary
print(f"📊 Analytics Summary:")
print(f"   Page Visits: {analytics['total_page_visits']}")
print(f"   Analyses: {analytics['total_analyses']}")
print(f"   Recent Activity: {len(analytics['recent_activity'])} events")

# Pretty print recent activity
print("\n🕒 Recent Activity:")
for activity in analytics['recent_activity'][-5:]:  # Last 5 events
    event_type = activity['event_type']
    timestamp = activity['timestamp']
    print(f"   {timestamp}: {event_type}")

# Use in Flask route
from flask import jsonify

@app.route("/dashboard")
def dashboard():
    analytics = get_analytics_summary()
    return jsonify(analytics)
