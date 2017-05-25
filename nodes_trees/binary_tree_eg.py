from treenode import Treenode
# create root
root = Treenode(8)
''' following is the tree after above statement
        8
      /   \
     None  None
'''
 
root.left      = Treenode(3);
root.right     = Treenode(10);
   
''' 3 and 10 become left and right children of 8
           8
         /   \
        3      10
     /    \    /  \
   None None None None'''
 
 
root.left.right  = Treenode(6);

'''6 becomes left child of 3
           8
       /       \
      3          10
    /   \       /  \
   None  6  None  None
'''
