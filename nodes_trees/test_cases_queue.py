from queue import Queue

qu = Queue()
assert qu.empty()==True
qu.push(1)
assert qu.empty()==False
print "Test #1 Passed"
qu.push(2)
qu.push(3)
qu.push(4)
assert qu.head.data==4
print "Test #2 Passed"
qu.pop()
assert qu.head.data==4
print "Test #3 Passed"
qu.pop()
assert qu.tail.data==3
print "Test #4 Passed"
