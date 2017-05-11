class Node: 
    def __init__(self,data, next=None, prev=None):
        self.data = data
        self.next = next
        self.prev = prev

    def __str__(self):
        return "Data: {}".format(self.data)

    def getData(self):
        return self.data

    def setData(self, new_data):
        self.data = new_data

    def getNext(self):
        return self.next

    def setNext(self, new_next):
        self.next = new_next

    def getPrev(self):
        return self.prev

    def setPrev(self, new_prev):
        self.prev = new_prev
