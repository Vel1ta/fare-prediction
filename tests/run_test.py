import pytest
import sys
import os

def main():
    repo_root = os.path.dirname(os.path.realpath(__file__))
    # Switch to the repository's root directory.
    os.chdir(repo_root)

    sys.dont_write_bytecode = True

    # Run pytest.
    retcode = pytest.main(sys.argv[1:])

    # Fail the cell execution if there are any test failures.
    assert retcode == 0, "The pytest invocation failed. See the log for details."

if __name__=="__main__":
    main()