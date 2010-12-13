#!/bin/sh

cd $1
echo $1
echo $1 > md5
for f in daetools*; do
  md5deep ${f} >> md5
done
