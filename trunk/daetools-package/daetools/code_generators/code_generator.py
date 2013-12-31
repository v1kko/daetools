from abc import ABCMeta, abstractmethod, abstractproperty

class daeCodeGenerator(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def generateSimulation(self, simulation, directory):
        pass
