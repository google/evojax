# EvoJAX Development Workflow

The EvoJAX development workflow is centralized on github repo, in the form of adding contribution through PR (Pull Request) or direct commitment to the repo. This part is not covered in this doc. Refer to Github docs [about pull request reviews](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/about-pull-request-reviews) for more information.
Also, we configure `CI/CD` as Github workflows [here](.github/workflows).

When we are doing a simple updating (e.g. simply updating `README.md` or adding a new example notebook)
- Simply commit and push.
- `[CI/CD Behavior]` When there is a push to **any** branch, CI/CD invokes `flake8` (for linting) and `pytest` (for testing, test cases under `./test`)

When we incorporated the contributions:
  - Bump version and commit: 
    - Edit the version specified on `evojax/version.py`. We follow the Package version schema in PEP 440, where as a minor we update the last digit, e.g. `0.2.0` -> `0.2.1`. 
    - Commit this change, e.g. `git commit -m <CommitMessage>` and push (don’t forget!) `git push origin`.
  - Add a git (lightweight) tag to the corresponding commit and push the tag to GitHub.
    - Pull the remote tags back to local git, e.g. git pull --tags just to ensure.
    - Ensure we are on the right commit.
    - Make a local git tag and push it to the remote. e.g. `git tag v0.2.1` and `git push origin --tags`. 
    - `[CI/CD Behavior]` When a tag is pushed, CI/CD will build the package and upload it to TestPyPI.
    - `[CI/CD Artifact]` we can test TestPyPI package with `pip install --index-url https://test.pypi.org/simple/ --no-deps --upgrade evojax`
  - Since we already have a tag, we create a "github release" corresponding to this tag on Github. 
    - `[CI/CD Behavior]` When a release is created, CI will build the package and upload it to PyPI.
    - `[CI/CD Artifact]` We can use the (public, official) PyPI package with `pip install evojax`.
