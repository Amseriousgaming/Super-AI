"""Makes `import app`, `import ai`, ... work when pytest is run from backend/."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
