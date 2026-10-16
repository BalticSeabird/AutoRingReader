# Event Logging System - Usage Guide

## Overview
A SQLite-based event logging system has been added to your ring detection pipeline. It automatically logs each video's processing results from all three models (ring detection, tag detection, and text recognition).

## Files Added/Modified

### New File: `event_logger.py`
- **EventLogger** class: Main logging interface
- Initializes SQLite database with schema on startup
- Provides methods to log different types of events

### Modified File: `ring_tag_read.py`
- Imports `EventLogger`
- Initializes database at `./log/events.db` when pipeline starts
- Logs video summary statistics after processing completes
- Logs are tied to the same directory as text logs

## Database Location
```
./log/events.db
```
The database is created automatically in the `log/` directory alongside your existing text logs.

## Database Schema

### Tables

#### 1. `events` (raw frame-level events)
- `id`: Primary key
- `timestamp`: When the event was logged
- `video_path`: Path to the video file
- `camera`: Camera name
- `frame_index`: Frame index within video
- `status`: Event status (success/error)

#### 2. `ring_detections` (ring detector outputs)
- `event_id`: Reference to events table
- `ring_conf`: Detection confidence
- `ring_x, ring_y, ring_w, ring_h`: Normalized bounding box coordinates
- `in_roi`: Whether detection is in region of interest

#### 3. `tag_detections` (tag detector outputs)
- `event_id`: Reference to events table
- `tag_count`: Number of tags detected
- `tag_conf_mean`: Mean confidence
- `tag_conf_max`: Maximum confidence

#### 4. `text_recognition` (OCR model outputs)
- `event_id`: Reference to events table
- `letters_str`: Recognized letters string
- `letters_conf_min`: Minimum confidence for letters
- `letters_conf_array`: Full confidence array (JSON)
- `numbers_str`: Recognized numbers string
- `numbers_conf_min`: Minimum confidence for numbers
- `numbers_conf_array`: Full confidence array (JSON)

#### 5. `video_summary` (one entry per video)
- `video_path`: Path to video (unique)
- `camera`: Camera name
- `processing_seconds`: Total processing time
- `frames_total`: Total frames processed
- `ring_detect_total`: Total ring detections
- `ring_detect_roi`: Detections in ROI
- `tag_detect_total`: Total tag detections
- `tag_detect_frames`: Frames with tags
- `ocr_final_accepts`: Final OCR accepts
- `final_tag`: Final recognized tag
- `final_tag_count`: Count of final tag
- `status`: Processing status (success/error)

## How to Query the Database

### Using Python
```python
import sqlite3

conn = sqlite3.connect('./log/events.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Get all video summaries
cursor.execute('SELECT * FROM video_summary')
for row in cursor.fetchall():
    print(dict(row))

# Get successful videos only
cursor.execute('SELECT * FROM video_summary WHERE status = ?', ('success',))

# Get videos by camera
cursor.execute('SELECT * FROM video_summary WHERE camera = ?', ('FAR6_SCALE',))

conn.close()
```

### Using SQLite CLI
```bash
# Open database
sqlite3 ./log/events.db

# View all videos
SELECT video_path, camera, final_tag, final_tag_count FROM video_summary;

# Statistics
SELECT camera, COUNT(*) as videos, AVG(processing_seconds) as avg_time
FROM video_summary GROUP BY camera;

# Find videos with detections
SELECT * FROM video_summary WHERE ring_detect_roi > 0 AND final_tag_count > 0;
```

## Automatic Logging

The logging happens automatically during pipeline execution:

1. **Pipeline Start**: `EventLogger` is initialized and database is created
2. **Video Processing**: Each video is processed by the multiprocessing pool
3. **Video Completion**: After all videos are processed, `log_video_summary()` is called for each video with:
   - Video metadata (path, camera)
   - Ring detection statistics
   - Tag detection statistics
   - OCR recognition statistics
   - Processing time and final results

## API Reference

### EventLogger Class

#### `__init__(db_path: Path)`
Initialize the logger with a database path.

#### `log_event(video_path, camera, frame_index=None, status='success') -> int`
Create a new base event and return its ID.

#### `log_ring_detection(event_id, ring_conf, ring_x, ring_y, ring_w, ring_h, in_roi)`
Log ring detector output for an event.

#### `log_tag_detection(event_id, tag_count, tag_conf_mean=None, tag_conf_max=None)`
Log tag detector output for an event.

#### `log_text_recognition(event_id, letters_str, letters_conf_min, letters_conf_array, numbers_str, numbers_conf_min, numbers_conf_array)`
Log text recognition output for an event.

#### `log_video_summary(video_path, camera, processing_seconds, frames_total, ring_detect_total, ring_detect_roi, tag_detect_total, tag_detect_frames, ocr_final_accepts, final_tag='', final_tag_count=0, status='success')`
Log the complete video processing summary (called automatically).

#### `query_events(video_path=None) -> list`
Query events from the database.

#### `get_video_summary(video_path) -> dict`
Get summary for a specific video.

## Example: Exporting to CSV

```python
import pandas as pd

df = pd.read_sql_query('SELECT * FROM video_summary',
                       sqlite3.connect('./log/events.db'))
df.to_csv('./log/video_summary.csv', index=False)
```

## Expanding the Logger

To log frame-level events (individual detections within videos), you can modify `_ring_tag_read()` to call:
- `log_event()` to create a frame event
- `log_ring_detection()` for ring detections
- `log_tag_detection()` for tag detections
- `log_text_recognition()` for OCR results

However, this requires access to the EventLogger instance in the multiprocessing workers (threading would be needed).
