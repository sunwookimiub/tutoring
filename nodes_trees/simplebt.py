"""
    tree eg
      ----
       1    <-- root
     /   \
    2     3 <-- leaf 
   /   
  4 <-- leaf
"""
from treenode import Treenode
# create root
root = Treenode(1)
''' following is the tree after above statement
        1
      /   \
     None  None'''
 
root.left      = Treenode(2);
root.right     = Treenode(3);
   
''' 2 and 3 become left and right children of 1
           1
         /   \
        2      3
     /    \    /  \
   None None None None'''
 
 
root.left.left  = Treenode(4);

'''4 becomes left child of 2
           1
       /       \
      2          3
    /   \       /  \
   4    None  None  None
  /  \
None None'''
