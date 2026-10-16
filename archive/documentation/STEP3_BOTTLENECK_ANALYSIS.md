# Step 3 Bottleneck Analysis - ROOT CAUSE IDENTIFIED

## Executive Summary

**Problem:** Step 3 got stuck because it loads ALL 27+ million crop files upfront before processing, causing extreme memory bloat and I/O thrashing.

**Timeline from logs:**
- Step 2 completed: Apr 24, 14:53:38
- Step 3 grouping started: 14:57:56
- **Time to load all crops: ~9 minutes** (just to build the dictionary!)
- Step 3 processing: 15:06:46 → Apr 25, 22:25:12 (**31+ hours** for OCR)

**Data scale:**
- Total Ring Crops: **27,131,518** (27+ million files!)
- Total Events: 13,651
- Total Tags: 50,100,336
- Avg crops per event: ~1,990 files

## Root Cause: `group_tag_crops_by_ring()` (line 36-71)

```python
def group_tag_crops_by_ring(output_root: str) -> Dict:
    tag_dir = Path(output_root) / "step2_tag_crops"
    groups = defaultdict(list)

    # ❌ BOTTLENECK: Recursively walks entire tree, sorts ALL files
    for crop_file in sorted(tag_dir.rglob("*.jpg")):  # Line 55
        parts = crop_file.parts
        camera = parts[-3]
        video_stem = parts[-2]
        stem = crop_file.stem
        if "_ring_tag_" in stem:
            frame_id = stem.split("_ring_tag_")[0]
            key = (camera, video_stem, frame_id)
            groups[key].append(str(crop_file))  # Add to dict entry

    # Sort each group by tag index
    for key in groups:
        groups[key] = sorted(groups[key])

    return dict(groups)  # Returns dict with 27M+ entries!
```

### Memory Impact
- **Dictionary overhead**: 27M+ keys + value lists
- **String storage**: File paths for 27M+ crop files (each ~100-200 bytes)
- **Estimated memory**: 30-50 GB just for the dictionary!
- **Plus runtime overhead**: Processing the data while this huge dict lives in memory

### Performance Impact
1. **9 minutes just to scan and sort** 27+ million files
2. **Massive I/O thrashing** as OS swaps this huge dict
3. **All GPU workers idle** waiting for the grouping to complete
4. **No streaming**: Everything processed at once instead of event-by-event

## Current Architecture vs. Optimal

### Step 2 (Working Well)
✅ Processes **events one-by-one** 
✅ Loads only **one event's crops** at a time (~1,990 files per event)
✅ Completes before next event loads
✅ Memory stays constant (~500MB-2GB per event)
✅ Can resume from any event

### Step 3 (Broken)
❌ Loads ALL 27M+ crops at once
❌ Builds giant 30-50GB dictionary
❌ Processes everything in one massive batch
❌ Memory bloat + I/O thrashing
❌ Cannot resume from individual events

## Solution: Event-by-Event Processing for Step 3

Instead of:
```python
tag_groups = group_tag_crops_by_ring(config.output_root_path)  # 27M files!
all_results = process_everything(tag_groups)
```

Do:
```python
for event in completed_events_from_step2:
    event_crops = group_crops_for_event(event)  # Only 1,990 files
    event_results = process_event(event_crops)
    save_event_results(event_results)
```

### Benefits
1. **Memory**: ~50-100x reduction (from 30GB → 500MB per event)
2. **I/O**: Streaming instead of thrashing
3. **Performance**: Faster (no giant sorting, no memory swapping)
4. **Resilience**: Can resume from any event
5. **Scalability**: Works with unlimited events (current approach can't scale beyond available RAM)

## Recommended Implementation

### Option A: Process Event-by-Event (Recommended)
- Iterate through step2_tag_crops directory
- For each event folder, group its crops
- Run OCR on that event
- Log to database
- Clear memory
- Move to next event

**Pros:** Simple, memory-efficient, matches Step 2 pattern, can resume
**Cons:** Slightly more I/O (but worth the tradeoff)

### Option B: Hybrid Batching
- Group crops by camera (4 cameras = 4 batches)
- Process one camera's data at a time (~6.8M crops each)
- Still 5-10x better than current approach

**Pros:** Balanced approach
**Cons:** Still uses significant memory, less resilient

## Estimated Fix Time
- Refactor `group_tag_crops_by_ring()` to iterate event-by-event: **1-2 hours**
- Update `_run_step3_single_gpu()` and `_run_step3_multi_gpu()` for event iteration: **30 min - 1 hour**
- Testing: **1-2 hours**
- **Total: 3-5 hours** (including testing)

## Next Steps
Ready to implement the event-by-event fix. This will:
1. Make Step 3 handle unlimited events without memory issues
2. Enable resume capability
3. Reduce wall-clock time (less memory swapping)
4. Match Step 2's robust design pattern
