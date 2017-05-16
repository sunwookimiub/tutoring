from node import Node
from testsinglylinkedlist import SinglyLinkedList
from testdoublylinkedlist import DoublyLinkedList
from linkedlist import LinkedList

# Test 2
dll = DoublyLinkedList()
assert dll.size()==0
dll.insert(1)
assert dll.size()==1
dll.insert(2)
assert dll.head.next.data == dll.tail.data
assert dll.head.data == dll.tail.prev.data
dll.insert(3)
assert dll.size()==3
print "Test 2 Passed"

# Test 3
dll = DoublyLinkedList()
assert dll.size()==0
dll.insert(1)
assert dll.search(1)==True
dll.insert(2)
dll.insert(3)
assert dll.head.next.data == 2
assert dll.search(3)==True
assert dll.search(4)==False
print "Test 3 Passed"

# Test 4
dll = DoublyLinkedList()
assert dll.size()==0
dll.insert(1)
dll.insert(2)
dll.insert(3)
dll.delete(2)
assert dll.tail.prev.data == 3
assert dll.search(2)==False
assert dll.search(3)==True
assert dll.size()==2
print "Test 4 Passed"

# Test 5
dll = DoublyLinkedList()
dll.insert(1)
dll.insert(2)
dll.insert(3)
dll.insert(4)
dll.print_list_forward()

# Test 6
dll = DoublyLinkedList()
dll.insert(1)
dll.insert(2)
dll.insert(3)
dll.insert(4)
dll.print_list_backward()
