from abc import ABCMeta, abstractmethod
from node import Node

class LinkedList():

    __metaclass__ = ABCMeta

    numLinks = 0

    def __init__(self, head=None, tail=None):
        self.head = head
        self.tail = tail

    @classmethod
    def getNumberOfLinks(cls):
        return cls.numLinks

    @abstractmethod
    def insert(self, data):
        pass

    @abstractmethod
    def size(self):
        pass

    @abstractmethod
    def search(self, data):
        pass

    @abstractmethod
    def delete(self, data):
        pass

    @abstractmethod
    def print_list_forward(self):
        pass

    @abstractmethod
    def print_list_backward(self):
        pass
