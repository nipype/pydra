# Contributing to Pydra

Welcome to the Pydra repository! We're excited you're here and want to contribute.

These guidelines are designed to make it as easy as possible to get involved.
If you have any questions that aren't discussed below, please let us know by opening an [issue][link_doc_issues]!

Before you start you'll need to set up a free GitHub account and sign in, here are some [instructions][link_signupinstructions].
If you are not familiar with version control systems such as git,
 introductions and tutorials may be found [here](http://www.reproducibleimaging.org/module-reproducible-basics/02-vcs/).


## How can you contribute

There are two main ways of contributing to the project:

**1. Provide suggestions, comments and report problems**

If you want to share anything with the community and developers, please open [a new issue][link_new_issues].
There are multiple templates that you can choose from, please fill them out to the best of your ability.
- **Bug report** - report if something is not working correctly or the documentation is incorrect
- **Documentation improvement** - request improvements to the documentation and tutorials
- **Feature request** - share an idea for a new feature, or changes to an existing feature
- **Maintenance and Delivery** - suggest changes to development infrastructure, testing, and delivery
- **Questions** - ask questions regarding the tool and the usage


**2. Improve the code and documentation**
We appreciate all improvement to the Pydra code or documentation.
Please try to follow the recommended steps, and don't hesitate to ask questions!


**i. Comment on an existing issue or open a new issue describing your idea**

This allows other members of the Pydra development team to confirm
that you aren't overlapping with work that's currently underway and
that everyone is on the same page with the goal of the work you're going to carry out.

**ii. [Fork][link_fork] the [Pydra repository][link_pydra] to your profile**

This is now your own unique copy of the Pydra repository.
Changes here won't affect anyone else's work, so it's a safe space to explore edits to the code!

**iii. [Clone][link_clone] the repository to your local machine**

You can clone your Pydra repository (from your fork!) in order to create a local copy of the code on your machine.

(Make sure to keep your fork up to date with the original Pydra repository.
One way to do this is to [configure a new remote named "upstream"](https://help.github.com/articles/configuring-a-remote-for-a-fork/)
 and to [sync your fork with the upstream repository][link_updateupstreamwiki].)


**iv. Install Pydra on your machine**

To install your version of Pydra, and the dependencies needed for development,
in your Python environment (Python 3.7 or higher), run `pip install -e ".[dev]"`
from your local Pydra directory.

In order to check if everything is working correctly, run the tests
using [pytest](https://docs.pytest.org/en/latest/), e.g. `pytest -vs pydra`

**v. Install pre-commit.**

[pre-commit](https://pre-commit.com/) is a git hook for running operations at commit time. To use it in
your environment, do `pip install pre-commit` following by `pre-commit install`
inside your source directory.


**vi. Make the changes you've discussed.**

It's a good practice to create [a new branch](https://help.github.com/articles/about-branches/)
of the repository for a new set of changes.
Once you start working on your changes, test frequently to ensure you are not breaking the existing code.
It's also a good idea to [commit][link_commit] your changes whenever
you finish specific task, and [push][link_push] your work to your GitHub repository.


**vii. Submit a [pull request][link_pullrequest].**

A new pull request for your changes should be created from your fork of the repository
after you push all the changes you made on your local machine.

When opening a pull request, please use one of the following prefixes:


* **[ENH]** for enhancements
* **[FIX]** for bug fixes
* **[MNT]** for maintenance
* **[TST]** for new or updated tests
* **[DOC]** for new or updated documentation
* **[STY]** for stylistic changes
* **[REF]** for refactoring existing code



**Pull requests should be submitted early and often (please don't mix too many unrelated changes within one PR)!**
If your pull request is not yet ready to be merged, please also include the **[WIP]** prefix (you can remove it once your PR is ready to be merged).
This tells the development team that your pull request is a "work-in-progress", and that you plan to continue working on it.

Review and discussion on new code can begin well before the work is complete, and the more discussion the better!
The development team may prefer a different path than you've outlined, so it's better to discuss it and get approval at the early stage of your work.

One your PR is ready a member of the development team will review your changes to confirm that they can be merged into the main codebase.

## Notes for New Code

#### Catching exceptions
In general, do not catch exceptions without good reason.
For non-fatal exceptions, log the exception as a warning and add more information about what may have caused the error.

If you do need to catch an exception, raise a new exception using ``raise NewException("message") from oldException)``.
Do not log this, as it creates redundant/confusing logs.

#### Testing
Testing is a crucial step of code development, remember:
- new code should be tested
- bug fixes should include an example that exposes the issue
- any new features should have tests that show at least a minimal example.

If you're not sure what this means for your code, please ask in your pull request,
we will help you with writing the tests.

## Recognizing contributions

We welcome and recognize all contributions from documentation to testing to code development.

The development team member who accepts/merges your pull request will update the CHANGES file to reference your contribution.

You can see a list of current contributors in our [zenodo file][link_zenodo].
If you are new to the project, don't forget to add your name and affiliation there!


[link_pydra]: https://github.com/nipype/pydra
[link_signupinstructions]: https://help.github.com/articles/signing-up-for-a-new-github-account
[link_new_issues]: https://github.com/nipype/pydra/issues/new/choose
[link_doc_issues]: https://github.com/nipype/pydra/issues/new?assignees=&labels=documentation&template=documentation.md&title=

[link_pullrequest]: https://help.github.com/articles/creating-a-pull-request-from-a-fork/
[link_fork]: https://help.github.com/articles/fork-a-repo/
[link_clone]: https://help.github.com/articles/cloning-a-repository/
[link_updateupstreamwiki]: https://help.github.com/articles/syncing-a-fork/
[link_push]: https://help.github.com/en/github/using-git/pushing-commits-to-a-remote-repository
[link_commit]: https://git-scm.com/docs/git-commit

[link_zenodo]: https://github.com/nipype/pydra/blob/master/.zenodo.json
