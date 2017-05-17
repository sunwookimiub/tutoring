from treenode import Treenode

# A function to do inorder tree traversal
def printInorder(root):
  if root:
    # First recur on left child
    printInorder(root.left)
    # then print the data of node
    print(root.data),
    # now recur on right child
    printInorder(root.right)

# A function to do postorder tree traversal
def printPostorder(root):
  if root:
    # First recur on left child
    printPostorder(root.left)
    # the recur on right child
    printPostorder(root.right)
    # now print the data of node
    print(root.data),

# A function to do postorder tree traversal
def printPreorder(root):
  if root:
    # First print the data of node
    print(root.data),
    # Then recur on left child
    printPreorder(root.left)
    # Finally recur on right child
    printPreorder(root.right)

# Driver code
root = Treenode(1)
root.left  = Treenode(2)
root.right   = Treenode(3)
root.left.left = Treenode(4)
root.left.right = Treenode(5)

print "Preorder traversal of binary tree is"
printPreorder(root)

print "\nInorder traversal of binary tree is"
printInorder(root)

print "\nPostorder traversal of binary tree is"
printPostorder(root)
