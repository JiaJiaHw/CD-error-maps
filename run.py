"""Source entry point; does not depend on the shell's current directory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cd_error_maps.cli import main

if __name__ == "__main__":
    sys.exit(main())
