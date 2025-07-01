"""
Date and time manipulation in Python can be done using the `datetime` module.
This module provides classes for manipulating dates and times, 
including formatting, parsing, and arithmetic operations.
"""

from datetime import datetime, timedelta


# 1. Get the current date and time
current_datetime = datetime.now()
print("Current date and time:", current_datetime)

# 2. Format date and time
formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
print("Formatted date and time:", formatted_datetime)

# 3. Parse a date string
date_string = "2023-10-01 12:30:45"
parsed_datetime = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
print("Parsed date and time:", parsed_datetime)

# 4. Date arithmetic
# Add 5 days to the current date
future_date = current_datetime + timedelta(days=5)
print("Future date (5 days later):", future_date)
# Subtract 3 hours from the current time
past_time = current_datetime - timedelta(hours=3)
print("Past time (3 hours earlier):", past_time)

# 5. Get the date part
date_part = current_datetime.date()
print("Date part:", date_part)

# 6. Get the time part
time_part = current_datetime.time()
print("Time part:", time_part)

# 7. Get the day of the week
day_of_week = current_datetime.strftime("%A")
print("Day of the week:", day_of_week)

# 8. Get the timestamp  # (seconds since epoch)
timestamp = current_datetime.timestamp()
print("Timestamp:", timestamp)

# 9. Convert timestamp back to datetime
converted_datetime = datetime.fromtimestamp(timestamp)
print("Converted datetime from timestamp:", converted_datetime)

# 10. Compare two dates
date1 = datetime(2023, 10, 1)
date2 = datetime(2023, 10, 15)
if date1 < date2:
    print(f"{date1} is earlier than {date2}")
else:
    print(f"{date1} is not earlier than {date2}")

# 11. Get the ISO format of the current date
iso_format = current_datetime.isoformat()
print("ISO format:", iso_format)

# 12. Get the UTC time
utc_datetime = datetime.utcnow()
print("Current UTC date and time:", utc_datetime)

# 13. Convert local time to UTC
utc_from_local = current_datetime.astimezone().astimezone(tz=None)
print("Converted local time to UTC:", utc_from_local)

# 14. Convert UTC time to local time
local_from_utc = utc_datetime.astimezone()
print("Converted UTC time to local time:", local_from_utc)

# 15. Create a specific date
specific_date = datetime(2023, 10, 1, 12, 30, 45)
print("Specific date and time:", specific_date)

# 16. Convert a date to a different timezone
from pytz import timezone
# Assuming pytz is installed, you can convert to a specific timezone
eastern = timezone('US/Eastern')
eastern_time = current_datetime.astimezone(eastern)
print("Current time in Eastern Time Zone:", eastern_time)

# 17. Get the difference between two dates
date1 = datetime(2023, 10, 1)
date2 = datetime(2023, 10, 15)
date_difference = date2 - date1
print("Difference between two dates:", date_difference.days, "days")