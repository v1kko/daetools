***************
Finite Volumes
***************
..
    Copyright (C) Dragan Nikolic
    DAE Tools is free software; you can redistribute it and/or modify it under the
    terms of the GNU General Public License version 3 as published by the Free Software
    Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE. See the GNU General Public License for more details.
    You should have received a copy of the GNU General Public License along with the
    DAE Tools software; if not, see <http://www.gnu.org/licenses/>.


Classes
=======
.. py:module:: daetools.pyDAE.hr_upwind_scheme
.. autosummary::
    :nosignatures:

    daeHRUpwindSchemeEquation

.. autoclass:: daeHRUpwindSchemeEquation
    :no-members:
    :no-undoc-members:

    .. autoattribute:: supported_flux_limiters
    .. automethod:: __init__
    .. automethod:: dc_dt
    .. automethod:: dc_dx
    .. automethod:: d2c_dx2
    .. automethod:: source

