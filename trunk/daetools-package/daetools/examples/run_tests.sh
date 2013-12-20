#!/bin/sh
# If called with no arguments the dwfault python is used
# Otherwise, the argument represents the python major.minor version

set -e

PYTHON=python$1
echo $PYTHON

echo whats_the_time.py
$PYTHON whats_the_time.py console

echo tutorial1.py
$PYTHON tutorial1.py console

echo tutorial2.py
$PYTHON tutorial2.py console

echo tutorial3.py
$PYTHON tutorial3.py console

echo tutorial4.py
$PYTHON tutorial4.py console

echo tutorial5.py
$PYTHON tutorial5.py console

echo tutorial6.py
$PYTHON tutorial6.py console

echo tutorial7.py
$PYTHON tutorial7.py console

echo tutorial8.py
$PYTHON tutorial8.py console

echo tutorial9.py
$PYTHON tutorial9.py console

echo tutorial10.py
$PYTHON tutorial10.py console

echo tutorial11.py
$PYTHON tutorial11.py console

echo tutorial12.py
$PYTHON tutorial12.py console

echo tutorial13.py
$PYTHON tutorial13.py console

echo tutorial14.py
$PYTHON tutorial14.py console

echo tutorial15.py
$PYTHON tutorial15.py console

echo tutorial16.py
$PYTHON tutorial16.py console

echo tutorial17.py
$PYTHON tutorial17.py console

echo tutorial18.py
$PYTHON tutorial18.py console

echo tutorial19.py
$PYTHON tutorial19.py console

echo tutorial_dealii_1.py
$PYTHON tutorial_dealii_1.py console

echo tutorial_dealii_2.py
$PYTHON tutorial_dealii_2.py console

echo opt_tutorial1.py
$PYTHON opt_tutorial1.py console

echo opt_tutorial2.py
$PYTHON opt_tutorial2.py console

echo opt_tutorial3.py
$PYTHON opt_tutorial3.py console

echo opt_tutorial4.py
$PYTHON opt_tutorial4.py console

echo opt_tutorial5.py
$PYTHON opt_tutorial5.py console

echo opt_tutorial6.py
$PYTHON opt_tutorial6.py console

echo opt_tutorial7.py
$PYTHON opt_tutorial7.py console
