# Criminal IP API Response Format Notes

This document contains information about the Criminal IP API response format changes and how the integration handles them.

## API Response Format Changes

The Criminal IP API has undergone some changes to its response format since the initial integration. The code has been updated to handle both the old and new formats.

### Key Changes:

1. **Domain Reports**:
   - Old format used `items` array in the response
   - New format uses `reports` array in the response

2. **Asset Search**:
   - Old format used `items` array in data
   - New format uses `result` array in data
   - Field naming changes for some properties

3. **Banner Search**:
   - Old format used `items` array and `total` count
   - New format uses `result` array and `count` for total

4. **IP Report**:
   - Response structure has been reorganized
   - Some field names have changed

## Compatibility Handling

The integration now handles both formats with graceful fallbacks:

```python
# Example for domain reports:
if "data" in results and "reports" in results["data"]:  
    items = results["data"]["reports"]
    count = results["data"].get("count", 0)

# Example for asset search:
if "data" in asset_results and "result" in asset_results["data"]:
    items = asset_results["data"]["result"]

# Example for banner search:
if "data" in results:
    # Check if "result" key exists (new API format)
    if "result" in results["data"]:
        items = results["data"]["result"]
        count = results["data"].get("count", 0)
    # Fall back to old API format if needed
    else:
        items = results["data"].get("items", [])
        count = results["data"].get("total", 0)
```

## Debugging Response Issues

When encountering issues with API responses:

1. Add debug output to examine the structure:
   ```python
   import json
   print(json.dumps(results, indent=2))
   ```

2. Check for any new field naming conventions

3. Update the corresponding formatter method in `criminalip_tool.py`

Remember to remove debug printouts after troubleshooting.
