from setuptools import setup, find_packages

setup(
    name="soccer_learning",
    version="0.0.1",
    author="Fan Jia",
    maintainer="nbro",
    author_email="",
    license="",
    description="A simple package that compares the performance and "
    "convergence of multiple learning algorithms (such as "
    "minimax Q-learning and Q-learning) on a simple soccer "
    "environment.",
    long_description=open("README.md").read(),
    packages=find_packages(exclude=["venv"]),
    install_requires=["numpy", "cvxopt", "matplotlib"],
    url="https://github.com/fjia30/MarkovSoccerGame",
    keywords=[
        "Reinforcement Learning",
        "Markov Decision Processes",
        "Multi-agent Reinforcement Learning",
        "Game Theory",
        "Markov Games",
        "Stochastic Games",
        "Minimax Q-Learning",
        "Foe-Q",
        "Friend-Q",
        "CEQ",
        "Q-Learning",
        "Soccer",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Researchers, Students, Scientists, Developers",
        "Topic :: Artificial Intelligence :: Reinforcement Learning",
        "Operating System :: MacOS :: Mac OS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Microsoft :: Linux",
        "Programming Language :: Python :: 3.9",
    ],
)
