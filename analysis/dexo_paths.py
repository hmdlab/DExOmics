"""dexo_paths.py — shared path resolution for the revision analysis scripts.

Resolution order (first hit wins):
  DATA_ROOT : --data-root <dir>  >  $DEXOMICS_DATA  >  ~/Data/DExOmics
  PROJ_ROOT : --proj-root <dir>  >  $DEXOMICS_PROJ  >  parent directory of this file (the repository root)

DATA_ROOT holds raw downloads (data/, gdc/, icgc/) and every intermediate/result (results/...).
PROJ_ROOT is only used for optional convenience copies of figures (notes/figures, TeX fig folders).
Both are exported to the environment so that subprocesses (Rscript, --part workers) inherit them.
The --data-root/--proj-root arguments are consumed here and removed from sys.argv so that the
scripts' own ad-hoc flag parsing never sees them.
"""
import os, sys


def _take(flag):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 >= len(sys.argv):
            sys.exit(f"{flag} requires a directory argument")
        val = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return val
    return None


DATA_ROOT = os.path.abspath(os.path.expanduser(_take("--data-root") or os.environ.get("DEXOMICS_DATA") or "~/Data/DExOmics"))
PROJ_ROOT = os.path.abspath(os.path.expanduser(_take("--proj-root") or os.environ.get("DEXOMICS_PROJ")
                                               or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["DEXOMICS_DATA"] = DATA_ROOT
os.environ["DEXOMICS_PROJ"] = PROJ_ROOT
if not os.path.isdir(DATA_ROOT):
    print(f"[dexo_paths] warning: DATA_ROOT {DATA_ROOT} does not exist yet", file=sys.stderr)
