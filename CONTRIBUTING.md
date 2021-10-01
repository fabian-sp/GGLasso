# Contributing to `GGLasso`

## Bug reports and feature requests

In case you have questions, possible bugs or feature requests, please, open an [issue](https://github.com/fabian-sp/GGLasso/issues) on Github. 

Pull requests will be evaluated against a checklist:

1.  __Motivation__. Your pull request should clearly and concisely motivate the need for change. Please, describe the problem your PR addresses and show how your pull request solves it as concisely as possible.

Also, include this motivation in `NEWS` so that when a new release of
`gglasso` comes out it is easy for users to see what has changed. Add your
item at the top of the file and use markdown for formatting. The
news item should end with `(@yourGithubUsername, #the_issue_number)`.

2.  __Only related changes__. Before you submit your pull request, please, check to make sure that you have NOT accidentally included any unrelated changes. These make it harder to see exactly what was changed, and to evaluate any unexpected side effects.

Each PR corresponds to a git branch, so if you expect to submit multiple changes
make sure to create multiple branches. If you have multiple changes that depend
on each other, start with the first one and do not submit any others until the
first one has been processed.

3. If you add new parameters or a new function, you will also need to document them.

## Contributing code

The procedure for contributing code is the following:

1. Fork the `GGLasso` repository into your own Github account (recommended: create a new branch for your feature).
2. Install the forked repository **from source** in developer mode (see the sections *Install from source* and *Advanced options* in our `README`).
3. Add your changes and commit them. 
4. Run `pytest gglasso/tests` to see whether all tests are successfull. If you add new functionalities, add unit tests for these.
5. Open a pull request for your feature branch on Github.

Also, have a look at [this guide on forking in Github](https://guides.github.com/activities/forking/).

Useful information on how to write bug reports and the principles of contributing to open-source code projects are described on [the scikit-learn webpage](https://scikit-learn.org/stable/developers/contributing.html).


## Fixing typos
You can fix typos, spelling mistakes, or grammatical errors in the documentation directly using the GitHub web interface, as long as the changes are made in the source file.

