import pytest
import sys
import os
import yaml

from tests.test_functions import test_positive_column_train
sys.dont_write_bytecode = True

def main():
    with open('../config.yaml') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    repo_name = config['model_name']

    # Get the path to this notebook, for example "/Workspace/Repos/{username}/{repo-name}".
    notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()

    # Get the repo's root directory name.
    repo_root = os.path.dirname(os.path.dirname(notebook_path))

    # Prepare to run pytest from the repo.
    os.chdir(f"/Workspace/{repo_root}/{repo_name}")
    print(os.getcwd())

    # Skip writing pyc files on a readonly filesystem.
    sys.dont_write_bytecode = True

    # Run pytest.
    retcode = pytest.main([".", "-v", "-p", "no:cacheprovider"])

    # Fail the cell execution if there are any test failures.
    assert retcode == 0, "The pytest invocation failed. See the log for details."

if __name__=="__main__":
    main()