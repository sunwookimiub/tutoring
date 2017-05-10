from abc import ABCMeta, abstractmethod
from node import Node

class LinkedList():

    __metaclass__ = ABCMeta

    numLinks = 0

    def __init__(self, head=None, prev=None):
        self.head = head
        self.prev = prev

    @classmethod
    def getNumberOfLinks(cls):
        print cls.numLinks

    @abstractmethod
    def insert(self, data):
        pass

    @abstractmethod
    def size(self):
        pass

    @abstractmethod
    def search(self):
        pass

    @abstractmethod
    def delete(self):
        pass

    @abstractmethod
    def print_list(self):
        pass
