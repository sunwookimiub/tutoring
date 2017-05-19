from node import Node
from singlylinkedlist import SinglyLinkedList
from doublylinkedlist import DoublyLinkedList
from linkedlist import LinkedList

# Test 2
sll = SinglyLinkedList()
assert sll.size()==0
sll.insert(1)
assert sll.size()==1
sll.insert(2)
sll.insert(3)
assert sll.size()==3
print "Test 2 Passed"

# Test 3
sll = SinglyLinkedList()
assert sll.size()==0
sll.insert(1)
assert sll.search(1)==True
sll.insert(2)
sll.insert(3)
assert sll.search(3)==True
assert sll.search(4)==False
print "Test 3 Passed"

# Test 4
sll = SinglyLinkedList()
assert sll.size()==0
sll.insert(1)
sll.insert(2)
sll.insert(3)
sll.delete(2)
assert sll.search(2)==False
assert sll.search(3)==True
assert sll.size()==2
print "Test 4 Passed"

# Test 5
sll = SinglyLinkedList()
sll.insert(1)
sll.insert(2)
sll.insert(3)
sll.insert(4)
sll.print_list_forward()

# Test 6
sll = SinglyLinkedList()
sll.insert(1)
sll.insert(2)
sll.insert(3)
sll.insert(4)
sll.print_list_backward()

# Test 1
assert 1 == SinglyLinkedList.getNumberOfLinks()
assert 0 == LinkedList.getNumberOfLinks()
sll = SinglyLinkedList()
assert 1 == sll.getNumberOfLinks()
print "Test 1 Passed"

