************************
Object-Oriented approach
************************
..
    Copyright (C) Dragan Nikolic, 2013
    DAE Tools is free software; you can redistribute it and/or modify it under the
    terms of the GNU General Public License version 3 as published by the Free Software
    Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE. See the GNU General Public License for more details.
    You should have received a copy of the GNU General Public License along with the
    DAE Tools software; if not, see <http://www.gnu.org/licenses/>.

The Object-Oriented approach to process modelling is adopted in **DAE Tools**.
The main characteristics of such an approach are:

* Everything is an object

* Models are classes derived from the base daeModel class

* Basically all OO concepts supported by the target language (c++, Python) are allowed,
  except few exceptions:
  * Multiple inheritance is supported
  * Models can be parametrized (using templates in c++)
  * Derived classes always inherit all declared parameters, variables, equations etc. (polymorphism achieved through virtual functions where the declaration takes place)
  * All parameters, variables, equations etc. remain public

* Hierarchical model decomposition


.. image:: http://sourceforge.net/apps/piwik/daetools/piwik.php?idsite=1&amp;rec=1&amp;url=wiki/
    :alt:
