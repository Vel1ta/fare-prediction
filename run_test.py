import pytest
import sys
import os

def main():
    sys.dont_write_bytecode = True

    # Run pytest.
    retcode = pytest.main(["tests/"])

    # Fail the cell execution if there are any test failures.
    assert retcode == 0, "The pytest invocation failed. See the log for details."

if __name__=="__main__":
    main()