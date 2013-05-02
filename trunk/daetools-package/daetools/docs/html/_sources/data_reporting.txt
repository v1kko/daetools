**********************
Module pyDataReporting
**********************

.. py:module:: pyDataReporting
.. py:currentmodule:: pyDataReporting
   
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
============================
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

DataReceiver classes
====================

.. autosummary::
    daeDataReceiver_t
    daeTCPIPDataReceiverServer
    
.. autoclass:: pyDataReporting.daeDataReceiver_t
    :members:
    :undoc-members:
        
.. autoclass:: pyDataReporting.daeTCPIPDataReceiverServer
    :members:
    :undoc-members:

DataReceiver data-containers
============================

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

.. autoclass:: pyDataReporting.daeDataReceiverProcess
    :members:
    :undoc-members:

    
.. image:: http://sourceforge.net/apps/piwik/daetools/piwik.php?idsite=1&amp;rec=1&amp;url=wiki/
    :alt:
