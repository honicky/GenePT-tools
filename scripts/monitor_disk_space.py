#!/usr/bin/env python
"""Simple disk space monitor for long-running data processing jobs."""

import shutil
import time
import sys
from pathlib import Path

def check_disk_space(path="/data", warn_threshold_gb=20, critical_threshold_gb=10):
  """Check available disk space and warn if running low."""
  total, used, free = shutil.disk_usage(path)
  free_gb = free / (1024**3)
  
  print(f"Disk space check for {path}:")
  print(f"  Free: {free_gb:.1f}GB")
  print(f"  Total: {total / (1024**3):.1f}GB")
  print(f"  Used: {used / (1024**3):.1f}GB ({used/total*100:.1f}%)")
  
  if free_gb < critical_threshold_gb:
    print(f"❌ CRITICAL: Only {free_gb:.1f}GB free! Consider stopping the job.")
    return False
  elif free_gb < warn_threshold_gb:
    print(f"⚠️  WARNING: Only {free_gb:.1f}GB free. Monitor closely.")
    return True
  else:
    print(f"✅ OK: {free_gb:.1f}GB available")
    return True

if __name__ == "__main__":
  path = sys.argv[1] if len(sys.argv) > 1 else "/data"
  check_disk_space(path)




