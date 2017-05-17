from stack import Stack

st = Stack()
assert st.empty()==True
st.push(1)
assert st.empty()==False
st.push(2)
st.push(3)
st.push(4)
assert st.head.data==4
st.pop()
assert st.head.data==3
st.pop()
assert st.head.data==2
print "All Tests Passed"
