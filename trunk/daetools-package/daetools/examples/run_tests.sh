#!/bin/sh
# If called with no arguments the default python is used
# Otherwise, the argument represents the user-defined python

set -e

PYTHON=$1
echo $PYTHON

ROOT="$( cd "$( dirname "$0" )" && pwd )"

echo whats_the_time.py
$PYTHON $ROOT/whats_the_time.py console

echo tutorial1.py
$PYTHON $ROOT/tutorial1.py console

echo tutorial2.py
$PYTHON $ROOT/tutorial2.py console

echo tutorial3.py
$PYTHON $ROOT/tutorial3.py console

echo tutorial4.py
$PYTHON $ROOT/tutorial4.py console

echo tutorial5.py
$PYTHON $ROOT/tutorial5.py console

echo tutorial6.py
$PYTHON $ROOT/tutorial6.py console

echo tutorial7.py
$PYTHON $ROOT/tutorial7.py console

echo tutorial8.py
$PYTHON $ROOT/tutorial8.py console

echo tutorial9.py
$PYTHON $ROOT/tutorial9.py console

echo tutorial10.py
$PYTHON $ROOT/tutorial10.py console

echo tutorial11.py
$PYTHON $ROOT/tutorial11.py console

echo tutorial12.py
$PYTHON $ROOT/tutorial12.py console

echo tutorial13.py
$PYTHON $ROOT/tutorial13.py console

echo tutorial14.py
# Does not work with compute stack
$PYTHON $ROOT/tutorial14.py console

echo tutorial15.py
$PYTHON $ROOT/tutorial15.py console

echo tutorial16.py
$PYTHON $ROOT/tutorial16.py console

echo tutorial17.py
# Problems on some platforms
$PYTHON $ROOT/tutorial17.py console

echo tutorial18.py
$PYTHON $ROOT/tutorial18.py console

echo tutorial19.py
# Does not work with compute stack
$PYTHON $ROOT/tutorial19.py console

echo tutorial20.py
$PYTHON $ROOT/tutorial20.py console

echo tutorial21.py
$PYTHON $ROOT/tutorial21.py console


echo tutorial_adv_1.py
$PYTHON $ROOT/tutorial_adv_1.py console

echo tutorial_adv_2.py
$PYTHON $ROOT/tutorial_adv_2.py console

echo tutorial_adv_3.py
$PYTHON $ROOT/tutorial_adv_3.py console

echo tutorial_adv_4.py
$PYTHON $ROOT/tutorial_adv_4.py console



echo tutorial_che_1.py
$PYTHON $ROOT/tutorial_che_1.py console

echo tutorial_che_2.py
$PYTHON $ROOT/tutorial_che_2.py console

echo tutorial_che_3.py
$PYTHON $ROOT/tutorial_che_3.py console

echo tutorial_che_4.py
# Does not work with compute stack
$PYTHON $ROOT/tutorial_che_4.py console

echo tutorial_che_5.py
# Does not work with compute stack
$PYTHON $ROOT/tutorial_che_5.py console

echo tutorial_che_6.py
# Does not work with compute stack
$PYTHON $ROOT/tutorial_che_6.py console

echo tutorial_che_7.py
$PYTHON $ROOT/tutorial_che_7.py console

echo tutorial_che_8.py
$PYTHON $ROOT/tutorial_che_8.py console

echo tutorial_che_9.py
$PYTHON $ROOT/tutorial_che_9.py console



echo tutorial_sa_1.py
$PYTHON $ROOT/tutorial_sa_1.py console

echo tutorial_sa_2.py
$PYTHON $ROOT/tutorial_sa_2.py console

echo tutorial_sa_3.py
$PYTHON $ROOT/tutorial_sa_3.py console



echo tutorial_cv_1.py
$PYTHON $ROOT/tutorial_cv_1.py console

echo tutorial_cv_2.py
$PYTHON $ROOT/tutorial_cv_2.py console

echo tutorial_cv_3.py
$PYTHON $ROOT/tutorial_cv_3.py console

echo tutorial_cv_4.py
$PYTHON $ROOT/tutorial_cv_4.py console

echo tutorial_cv_5.py
$PYTHON $ROOT/tutorial_cv_5.py console

echo tutorial_cv_6.py
$PYTHON $ROOT/tutorial_cv_6.py console

echo tutorial_cv_7.py
$PYTHON $ROOT/tutorial_cv_7.py console

echo tutorial_cv_8.py
$PYTHON $ROOT/tutorial_cv_8.py console

echo tutorial_cv_9.py
$PYTHON $ROOT/tutorial_cv_9.py console

echo tutorial_cv_10.py
$PYTHON $ROOT/tutorial_cv_10.py console

echo tutorial_cv_11.py
$PYTHON $ROOT/tutorial_cv_11.py console



echo tutorial_dealii_1.py
$PYTHON $ROOT/tutorial_dealii_1.py console

echo tutorial_dealii_2.py
$PYTHON $ROOT/tutorial_dealii_2.py console

echo tutorial_dealii_3.py
$PYTHON $ROOT/tutorial_dealii_3.py console

echo tutorial_dealii_4.py
$PYTHON $ROOT/tutorial_dealii_4.py console

echo tutorial_dealii_5.py
$PYTHON $ROOT/tutorial_dealii_5.py console

echo tutorial_dealii_5.py
$PYTHON $ROOT/tutorial_dealii_5.py console

echo tutorial_dealii_6.py
$PYTHON $ROOT/tutorial_dealii_6.py console

echo tutorial_dealii_7.py
$PYTHON $ROOT/tutorial_dealii_7.py console

echo tutorial_dealii_8.py
$PYTHON $ROOT/tutorial_dealii_8.py console



echo opt_tutorial1.py
$PYTHON $ROOT/opt_tutorial1.py console

echo opt_tutorial2.py
$PYTHON $ROOT/opt_tutorial2.py console

echo opt_tutorial3.py
$PYTHON $ROOT/opt_tutorial3.py console

echo opt_tutorial4.py
$PYTHON $ROOT/opt_tutorial4.py console

echo opt_tutorial5.py
$PYTHON $ROOT/opt_tutorial5.py console

echo opt_tutorial6.py
$PYTHON $ROOT/opt_tutorial6.py console

echo opt_tutorial7.py
$PYTHON $ROOT/opt_tutorial7.py console



echo tutorial_che_opt_1.py
#$PYTHON $ROOT/tutorial_che_opt_1.py console

echo tutorial_che_opt_2.py
#$PYTHON $ROOT/tutorial_che_opt_2.py console

echo tutorial_che_opt_3.py
#$PYTHON $ROOT/tutorial_che_opt_3.py console

echo tutorial_che_opt_4.py
$PYTHON $ROOT/tutorial_che_opt_4.py console

echo tutorial_che_opt_5.py
$PYTHON $ROOT/tutorial_che_opt_5.py console

echo tutorial_che_opt_6.py
$PYTHON $ROOT/tutorial_che_opt_6.py console
