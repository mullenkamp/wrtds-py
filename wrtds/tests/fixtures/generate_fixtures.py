"""Generate R EGRET reference fixtures by calling Rscript via subprocess.

Can be run standalone::

    python -m wrtds.tests.fixtures.generate_fixtures

Or programmatically::

    from wrtds.tests.fixtures.generate_fixtures import generate
    generate()
"""

import shutil
import subprocess
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent
R_SCRIPT = FIXTURE_DIR / 'generate_fixtures.R'

EXPECTED_FILES = [
    'choptank_daily_input.csv',
    'choptank_sample_input.csv',
    'choptank_info.json',
    'choptank_daily_fitted.csv',
    'choptank_sample_cv.csv',
    'choptank_surfaces.bin',
    'choptank_surface_index.json',
    'choptank_annual.csv',
    'choptank_daily_kalman.csv',
    'choptank_pairs.csv',
    'choptank_groups.csv',
]


def generate(output_dir=None):
    """Run the R fixture generation script.

    Args:
        output_dir: Directory for output files.  Defaults to the fixtures
            directory alongside this module.

    Raises:
        RuntimeError: If Rscript is not found or the R script fails.
        FileNotFoundError: If expected output files are missing after generation.
    """
    if output_dir is None:
        output_dir = FIXTURE_DIR

    output_dir = Path(output_dir)

    rscript = shutil.which('Rscript')
    if rscript is None:
        raise RuntimeError(
            'Rscript not found on PATH. Install R and the EGRET package '
            'to regenerate fixtures.'
        )

    result = subprocess.run(
        [rscript, str(R_SCRIPT), str(output_dir)],
        capture_output=True,
        text=True,
        timeout=600,  # 10 minutes
    )

    if result.returncode != 0:
        raise RuntimeError(
            f'R fixture generation failed (exit code {result.returncode}):\n'
            f'stdout:\n{result.stdout}\n'
            f'stderr:\n{result.stderr}'
        )

    # Print R output for visibility
    if result.stdout:
        print(result.stdout)

    # Verify all expected files exist
    missing = [f for f in EXPECTED_FILES if not (output_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f'R script completed but missing expected files: {missing}'
        )


if __name__ == '__main__':
    generate()
