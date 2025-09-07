from config import validate_config
import sys

if __name__ == "__main__":
    try:
        validate_config()
        sys.exit(0)
    except Exception:
        sys.exit(1)