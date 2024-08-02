# reference: week 5 course note
# https://henryiii.github.io/se-for-sci/content/week05/task_runners.html
import nox


@nox.session
def tests(session):
    session.install("-e.[test]")
    session.run("pytest")


@nox.session
def docs(session):
    session.install("-e.[docs]")
    session.chdir("docs")
    session.run("sphinx-build", "-M", "html", ".", "_build")


@nox.session
def doc(session):
    session.install("sphinx")
    session.run("sphinx-build", "-b", "html", "docs/source", "docs/build")
