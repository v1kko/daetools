#!/bin/sh
cd /home/ciroki/Data/daetools/trunk/Website/api_ref
"Results" > results
pydoc -w core >> results
pydoc -w activity >> results
pydoc -w solver >> results
pydoc -w datareporting >> results
pydoc -w logging >> results
