from treenode import Treenode
# create root
root = Treenode(2)
''' following is the tree after above statement
        2
      /   \
     None  None
'''
 
root.right = Treenode(5);
   
''' 5 become right children of 2
           2
         /   \
      None    5
             /  \
           None None
'''
