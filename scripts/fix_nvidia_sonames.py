"""Create missing soname symlinks for nvidia pip packages.

The pip-managed nvidia packages (e.g. nvidia-cuda-nvrtc==13.0.88) sometimes
ship only the fully-versioned .so (libnvrtc.so.13.0.88) and an unversioned
.so (libnvrtc.so), but NOT the soname (libnvrtc.so.13) that libraries like
torchcodec link against.  This script creates those missing symlinks.
"""

import os
import pathlib
import re
import sys

import nvidia

nv_root = pathlib.Path(nvidia.__path__[0])
lib_dirs = {str(p.parent) for p in nv_root.rglob("lib/libnv*.so*")}
created = 0

for d in lib_dirs:
    for p in pathlib.Path(d).glob("*.so.*.*"):
        m = re.match(r"(.+\.so\.\d+)\.\d+.*", p.name)
        if m:
            soname = pathlib.Path(d) / m.group(1)
            if not soname.exists():
                os.symlink(p.name, str(soname))
                print(f"  Created soname symlink: {soname.name} -> {p.name}", file=sys.stderr)
                created += 1

print(f"Created {created} soname symlinks", file=sys.stderr)
