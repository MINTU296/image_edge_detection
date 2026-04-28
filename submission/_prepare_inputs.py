"""Stage input images for the demo run.

Copies the original photo at three resolutions (256, 1024, 4096) so the run
covers both small and large data. Names use the `before_` prefix to match
the `after_` output convention.
"""
import shutil
import sys
from pathlib import Path
from PIL import Image

src = Path(sys.argv[1])
inputs_dir = Path(sys.argv[2])
inputs_dir.mkdir(parents=True, exist_ok=True)

img = Image.open(src).convert('RGB')

shutil.copy2(src, inputs_dir / "before_01_original_BenTennyson.jpg")

for size in (256, 1024, 4096):
    out = inputs_dir / f"before_02_BenTennyson_{size}x{size}.png"
    img.resize((size, size), Image.LANCZOS).save(out, optimize=True)

print("Staged:")
for p in sorted(inputs_dir.iterdir()):
    print(f"  {p.name}  ({p.stat().st_size/1024:.1f} KB)")
