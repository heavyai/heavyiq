import json
import sys

import vl_convert as vlc

if __name__ == "__main__":
    spec = json.load(sys.stdin)
    png = vlc.vegalite_to_png(vl_spec=spec, vl_version="v5.15", scale=2)
    sys.stdout.buffer.write(png)
