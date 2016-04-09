**********************
Module pyDataReporting
**********************
..
    Copyright (C) Dragan Nikolic, 2016
    DAE Tools is free software; you can redistribute it and/or modify it under the
    terms of the GNU General Public License version 3 as published by the Free Software
    Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE. See the GNU General Public License for more details.
    You should have received a copy of the GNU General Public License along with the
    DAE Tools software; if not, see <http://www.gnu.org/licenses/>.

.. py:module:: pyDataReporting
   
Overview
==========


DataReporter classes
====================

.. autosummary::
    daeDataReporter_t
    daeDataReporterLocal
    daeNoOpDataReporter
    daeDataReporterFile
    daeTEXTFileDataReporter
    daeBlackHoleDataReporter
    daeDelegateDataReporter

.. autoclass:: pyDataReporting.daeDataReporter_t
    :members:
    :undoc-members:
    :exclude-members: Connect, Disconnect, IsConnected, StartRegistration, RegisterDomain, RegisterVariable,
                      EndRegistration, StartNewResultSet, EndOfData, SendVariable

    .. method:: Connect((daeDataReporter_t)self, (str)connectionString, (str)processName) -> bool
    .. method:: Disconnect((daeDataReporter_t)self) -> bool
    .. method:: IsConnected((daeDataReporter_t)self) -> bool
    .. method:: StartRegistration((daeDataReporter_t)self) -> bool
    .. method:: RegisterDomain((daeDataReporter_t)self, (daeDataReporterDomain)domain) -> bool
    .. method:: RegisterVariable((daeDataReporter_t)self, (daeDataReporterVariable)variable) -> bool
    .. method:: EndRegistration((daeDataReporter_t)self) -> bool
    .. method:: StartNewResultSet((daeDataReporter_t)self, (Float)time) -> bool
    .. method:: SendVariable((daeDataReporter_t)self, (daeDataReporterVariableValue)variableValue) -> bool
    .. method:: EndOfData((daeDataReporter_t)self) -> bool
        
Data reporters that *do not* send data to a data receiver and keep data locally (*local data reporters*)
--------------------------------------------------------------------------------------------------------

.. autoclass:: pyDataReporting.daeDataReporterLocal
    :members:
    :undoc-members:
    :exclude-members: Connect, Disconnect, IsConnected, StartRegistration, RegisterDomain, RegisterVariable,
                      EndRegistration, StartNewResultSet, EndOfData, SendVariable

.. autoclass:: pyDataReporting.daeNoOpDataReporter
    :members:
    :undoc-members:
    :exclude-members: Connect, Disconnect, IsConnected, StartRegistration, RegisterDomain, RegisterVariable,
                      EndRegistration, StartNewResultSet, EndOfData, SendVariable

.. autoclass:: pyDataReporting.daeDataReporterFile
    :members:
    :undoc-members:
    :exclude-members: Connect, Disconnect, IsConnected, StartRegistration, RegisterDomain, RegisterVariable,
                      EndRegistration, StartNewResultSet, EndOfData, SendVariable,
                      WriteDataToFile
                      
    .. method:: WriteDataToFile((daeDataReporterFile)self) -> None
        
.. autoclass:: pyDataReporting.daeTEXTFileDataReporter
    :members:
    :undoc-members:
    :exclude-members: Connect, Disconnect, IsConnected, StartRegistration, RegisterDomain, RegisterVariable,
                      EndRegistration, StartNewResultSet, EndOfData, SendVariable,
                      WriteDataToFile

    .. method:: WriteDataToFile((daeTEXTFileDataReporter)self) -> None

Third-party local data reporters
--------------------------------
.. py:module:: daetools.pyDAE.data_reporters

.. autosummary::
    daePlotDataReporter
    daeMatlabMATFileDataReporter
    daeExcelFileDataReporter
    daeJSONFileDataReporter
    daeXMLFileDataReporter
    daeHDF5FileDataReporter
    daePandasDataReporter

.. autoclass:: daetools.pyDAE.data_reporters.daePlotDataReporter
    :members:
    :undoc-members:
    :exclude-members: Connect, Disconnect, IsConnected, StartRegistration, RegisterDomain, RegisterVariable,
                      EndRegistration, StartNewResultSet, EndOfData, SendVariable

.. autoclass:: daetools.pyDAE.data_reporters.daeMatlabMATFileDataReporter
    :members:
    :undoc-members:
    :exclude-members: Connect, Disconnect, IsConnected, StartRegistration, RegisterDomain, RegisterVariable,
                      EndRegistration, StartNewResultSet, EndOfData, SendVariable

.. autoclass:: daetools.pyDAE.data_reporters.daeExcelFileDataReporter
    :members:
    :undoc-members:
    :exclude-members: Connect, Disconnect, IsConnected, StartRegistration, RegisterDomain, RegisterVariable,
                      EndRegistration, StartNewResultSet, EndOfData, SendVariable

.. autoclass:: daetools.pyDAE.data_reporters.daeJSONFileDataReporter
    :members:
    :undoc-members:
    :exclude-members: Connect, Disconnect, IsConnected, StartRegistration, RegisterDomain, RegisterVariable,
                      EndRegistration, StartNewResultSet, EndOfData, SendVariable

.. autoclass:: daetools.pyDAE.data_reporters.daeXMLFileDataReporter
    :members:
    :undoc-members:
    :exclude-members: Connect, Disconnect, IsConnected, StartRegistration, RegisterDomain, RegisterVariable,
                      EndRegistration, StartNewResultSet, EndOfData, SendVariable

.. autoclass:: daetools.pyDAE.data_reporters.daeHDF5FileDataReporter
    :members:
    :undoc-members:
    :exclude-members: Connect, Disconnect, IsConnected, StartRegistration, RegisterDomain, RegisterVariable,
                      EndRegistration, StartNewResultSet, EndOfData, SendVariable

.. autoclass:: daetools.pyDAE.data_reporters.daePandasDataReporter
    :members:
    :undoc-members:
    :exclude-members: Connect, Disconnect, IsConnected, StartRegistration, RegisterDomain, RegisterVariable,
                      EndRegistration, StartNewResultSet, EndOfData, SendVariable


.. py:currentmodule:: pyDataReporting

Data reporters that *do* send data to a data receiver (*remote data reporters*)
-------------------------------------------------------------------------------
.. autoclass:: pyDataReporting.daeDataReporterRemote
    :members:
    :undoc-members:
    :exclude-members: Connect, Disconnect, IsConnected, StartRegistration, RegisterDomain, RegisterVariable,
                      EndRegistration, StartNewResultSet, EndOfData, SendVariable,
                      SendMessage
                      
    .. method:: SendMessage((daeDataReporterRemote)self, (str)message) -> bool

.. autoclass:: pyDataReporting.daeTCPIPDataReporter
    :members:
    :undoc-members:
    :exclude-members: Connect, Disconnect, IsConnected, StartRegistration, RegisterDomain, RegisterVariable,
                      EndRegistration, StartNewResultSet, EndOfData, SendVariable,
                      SendMessage

    .. method:: SendMessage((daeTCPIPDataReporter)self, (str)message) -> bool

    
Special-purpose data reporters
------------------------------
.. autoclass:: pyDataReporting.daeBlackHoleDataReporter
    :members:
    :undoc-members:
    :exclude-members: Connect, Disconnect, IsConnected, StartRegistration, RegisterDomain, RegisterVariable,
                      EndRegistration, StartNewResultSet, EndOfData, SendVariable

    Data reporter that does not process any data and all function calls simply return ``True``.
    Could be used when no results from the simulation are needed.


.. autoclass:: pyDataReporting.daeDelegateDataReporter
    :members:
    :undoc-members:
    :exclude-members: Connect, Disconnect, IsConnected, StartRegistration, RegisterDomain, RegisterVariable,
                      EndRegistration, StartNewResultSet, EndOfData, SendVariable

    A container-like data reporter, which does not process any data but forwards (delegates) all function calls
    (:py:meth:`~pyDataReporting.daeDataReporter_t.Disconnect`, :py:meth:`~pyDataReporting.daeDataReporter_t.IsConnected`,
    :py:meth:`~pyDataReporting.daeDataReporter_t.StartRegistration`, :py:meth:`~pyDataReporting.daeDataReporter_t.RegisterDomain`,
    :py:meth:`~pyDataReporting.daeDataReporter_t.RegisterVariable`, :py:meth:`~pyDataReporting.daeDataReporter_t.EndRegistration`,
    :py:meth:`~pyDataReporting.daeDataReporter_t.StartNewResultSet`, :py:meth:`~pyDataReporting.daeDataReporter_t.SendVariable`,
    :py:meth:`~pyDataReporting.daeDataReporter_t.EndOfData`) to data reporters in the containing list of data reporters.
    Data reporters can be added by using the :py:meth:`~pyDataReporting.daeDataReporter_t.AddDataReporter`.
    The list of containing data reporters is in the :py:attr:`~pyDataReporting.daeBlackHoleDataReporter.DataReporters`
    attribute.

    .. method:: Connect((daeDataReporter_t)self, (str)connectionString, (str)processName) -> Boolean

        Does nothing. Always returns ``True``.

DataReporter data-containers
----------------------------
.. autosummary::
    daeDataReporterDomain
    daeDataReporterVariable
    daeDataReporterVariableValue

.. autoclass:: pyDataReporting.daeDataReporterDomain
    :members:
    :undoc-members:
    
.. autoclass:: pyDataReporting.daeDataReporterVariable
    :members:
    :undoc-members:
        
.. autoclass:: pyDataReporting.daeDataReporterVariableValue
    :members:
    :undoc-members:

    .. automethod:: __getitem__
    .. automethod:: __setitem__

DataReceiver classes
====================

.. autosummary::
    daeDataReceiver_t
    daeTCPIPDataReceiver
    daeTCPIPDataReceiverServer
    
.. autoclass:: pyDataReporting.daeDataReceiver_t
    :members:
    :undoc-members:
    :exclude-members: Start, Stop, Process

    .. method:: Start((daeDataReceiver_t)self) -> bool

    .. method:: Stop((daeDataReceiver_t)self) -> bool

    .. attribute:: Process


        
.. autoclass:: pyDataReporting.daeTCPIPDataReceiver
    :members:
    :undoc-members:
    :exclude-members: Start, Stop, Process

    .. automethod:: Start
    .. automethod:: Stop
    .. autoattribute:: Process

.. autoclass:: pyDataReporting.daeTCPIPDataReceiverServer
    :members:
    :undoc-members:

DataReceiver data-containers
----------------------------

.. autosummary::
    daeDataReceiverDomain
    daeDataReceiverVariable
    daeDataReceiverVariableValue
    daeDataReceiverProcess

.. autoclass:: pyDataReporting.daeDataReceiverDomain
    :members:
    :undoc-members:

.. autoclass:: pyDataReporting.daeDataReceiverVariable
    :members:
    :undoc-members:

.. autoclass:: pyDataReporting.daeDataReceiverVariableValue
    :members:
    :undoc-members:

    .. automethod:: __getitem__
    .. automethod:: __setitem__

.. autoclass:: pyDataReporting.daeDataReceiverProcess
    :members:
    :undoc-members:

    
