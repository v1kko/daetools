from libdaepython import *

# DataReporter from scratch
# All functions have to be overridden
class MyDataReporter(daeDataReporter_t):
	def __init__(self):
		daeDataReporter_t.__init__(self)

	def Connect(self, strConnectString, strProcessName):
		print "MyDataReporter Connect called"
		return True
		
	def Disconnect(self):
		print "MyDataReporter Disconnect called"
		return True

	def IsConnected(self):
		print "MyDataReporter IsConnected called"
		return True

	def StartRegistration(self):
		print "MyDataReporter StartRegistration called"
		return True

	def RegisterDomain(self, domain):
		print "MyDataReporter RegisterDomain called"
		return True

	def RegisterVariable(self, var):
		print "MyDataReporter RegisterVariable called"
		return True

	def EndRegistration(self):
		print "MyDataReporter EndRegistration called"
		return True

	def StartNewResultSet(self, time):
		print "MyDataReporter StartNewResultSet called"
		print "Time =", time
		return True

	def EndOfData(self):
		print "MyDataReporter EndOfData called"
		return True

	def SendVariable(self, var):
		print "MyDataReporter SendVariable called"
		return True

# DataReporterLocal
# Connect, Disconnect and IsConnected have to be overridden
class MyDataReporterLocal(daeDataReporterLocal):
	def __init__(self):
		daeDataReporterLocal.__init__(self)

	def Connect(self, strConnectString, strProcessName):
		print "MyDataReporterLocal Connect called"
		return True
		
	def Disconnect(self):
		print "MyDataReporterLocal Disconnect called"
		return True
		
	def IsConnected(self):
		print "MyDataReporterLocal IsConnected called"
		return True

	
# daeDataReporterFile
# Only WriteDataToFile has to be overridden
class MyDataReporterFile(daeDataReporterFile):
	def __init__(self):
		daeDataReporterFile.__init__(self)
	
	def WriteDataToFile(self):
		print "MyDataReporterFile WriteDataToFile called"

# daeDataReporterRemote
# Connect, Disconnect, IsConnected and SendMessage have to be overridden
class MyDataReporterRemote(daeDataReporterRemote):
	def __init__(self):
		daeDataReporterRemote.__init__(self)
	
	def Connect(self, strConnectString, strProcessName):
		print "MyDataReporterRemote Connect called"
		return True
		
	def Disconnect(self):
		print "MyDataReporterRemote Disconnect called"
		return True
		
	def IsConnected(self):
		print "MyDataReporterRemote IsConnected called"
		return True

	def SendMessage(self, message):
		print "MyDataReporterRemote SendMessage called"
		#print message
		return True

# daeDataReceiver_t
# Start, Stop and GetProcess have to be overridden
class MyDataReceiver(daeDataReceiver_t):
	def __init__(self):
		daeDataReceiver_t.__init__(self)

	def Start(self):
		print "MyDataReceiver Start called"
		return True
	
	def Stop(self):
		print "MyDataReceiver Stop called"
		return True

	def GetProcess(self):
		print "MyDataReceiver GetProcess called"
		return None

