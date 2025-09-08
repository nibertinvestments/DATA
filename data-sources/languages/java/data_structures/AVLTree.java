/**
 * Advanced AVL Tree Implementation
 * 
 * A self-balancing binary search tree where the heights of the two child
 * subtrees of any node differ by at most one. This implementation includes
 * comprehensive rotation methods and maintains balance through insertions
 * and deletions.
 * 
 * Time Complexities:
 * - Search: O(log n)
 * - Insert: O(log n)
 * - Delete: O(log n)
 * 
 * Space Complexity: O(n)
 * 
 * @author AI Training Dataset
 * @version 1.0
 */
public class AVLTree<T extends Comparable<T>> {
    
    /**
     * Node class representing each element in the AVL tree
     */
    private class Node {
        T data;
        Node left, right;
        int height;
        
        public Node(T data) {
            this.data = data;
            this.height = 1;
        }
    }
    
    private Node root;
    private int size;
    
    /**
     * Default constructor
     */
    public AVLTree() {
        this.root = null;
        this.size = 0;
    }
    
    /**
     * Get the height of a node
     * @param node The node to get height for
     * @return Height of the node (0 for null nodes)
     */
    private int getHeight(Node node) {
        return node == null ? 0 : node.height;
    }
    
    /**
     * Calculate the balance factor of a node
     * @param node The node to calculate balance for
     * @return Balance factor (left height - right height)
     */
    private int getBalance(Node node) {
        return node == null ? 0 : getHeight(node.left) - getHeight(node.right);
    }
    
    /**
     * Update the height of a node based on its children
     * @param node The node to update
     */
    private void updateHeight(Node node) {
        if (node != null) {
            node.height = 1 + Math.max(getHeight(node.left), getHeight(node.right));
        }
    }
    
    /**
     * Perform right rotation
     * @param y The node to rotate around
     * @return New root after rotation
     */
    private Node rotateRight(Node y) {
        Node x = y.left;
        Node T2 = x.right;
        
        // Perform rotation
        x.right = y;
        y.left = T2;
        
        // Update heights
        updateHeight(y);
        updateHeight(x);
        
        return x;
    }
    
    /**
     * Perform left rotation
     * @param x The node to rotate around
     * @return New root after rotation
     */
    private Node rotateLeft(Node x) {
        Node y = x.right;
        Node T2 = y.left;
        
        // Perform rotation
        y.left = x;
        x.right = T2;
        
        // Update heights
        updateHeight(x);
        updateHeight(y);
        
        return y;
    }
    
    /**
     * Insert a value into the AVL tree
     * @param data The value to insert
     */
    public void insert(T data) {
        root = insertHelper(root, data);
    }
    
    /**
     * Recursive helper method for insertion
     * @param node Current node in traversal
     * @param data Data to insert
     * @return Updated node after insertion and balancing
     */
    private Node insertHelper(Node node, T data) {
        // Standard BST insertion
        if (node == null) {
            size++;
            return new Node(data);
        }
        
        int cmp = data.compareTo(node.data);
        if (cmp < 0) {
            node.left = insertHelper(node.left, data);
        } else if (cmp > 0) {
            node.right = insertHelper(node.right, data);
        } else {
            // Duplicate values not allowed
            return node;
        }
        
        // Update height
        updateHeight(node);
        
        // Get balance factor
        int balance = getBalance(node);
        
        // Perform rotations if needed
        
        // Left Left Case
        if (balance > 1 && data.compareTo(node.left.data) < 0) {
            return rotateRight(node);
        }
        
        // Right Right Case
        if (balance < -1 && data.compareTo(node.right.data) > 0) {
            return rotateLeft(node);
        }
        
        // Left Right Case
        if (balance > 1 && data.compareTo(node.left.data) > 0) {
            node.left = rotateLeft(node.left);
            return rotateRight(node);
        }
        
        // Right Left Case
        if (balance < -1 && data.compareTo(node.right.data) < 0) {
            node.right = rotateRight(node.right);
            return rotateLeft(node);
        }
        
        return node;
    }
    
    /**
     * Delete a value from the AVL tree
     * @param data The value to delete
     * @return true if deleted, false if not found
     */
    public boolean delete(T data) {
        int initialSize = size;
        root = deleteHelper(root, data);
        return size < initialSize;
    }
    
    /**
     * Recursive helper method for deletion
     * @param node Current node in traversal
     * @param data Data to delete
     * @return Updated node after deletion and balancing
     */
    private Node deleteHelper(Node node, T data) {
        if (node == null) {
            return node;
        }
        
        int cmp = data.compareTo(node.data);
        if (cmp < 0) {
            node.left = deleteHelper(node.left, data);
        } else if (cmp > 0) {
            node.right = deleteHelper(node.right, data);
        } else {
            // Node to be deleted found
            size--;
            
            // Node with only one child or no child
            if (node.left == null || node.right == null) {
                Node temp = (node.left != null) ? node.left : node.right;
                
                if (temp == null) {
                    // No child case
                    temp = node;
                    node = null;
                } else {
                    // One child case
                    node = temp;
                }
            } else {
                // Node with two children
                Node temp = findMin(node.right);
                node.data = temp.data;
                node.right = deleteHelper(node.right, temp.data);
                size++; // Compensate for the decrement in recursive call
            }
        }
        
        if (node == null) {
            return node;
        }
        
        // Update height
        updateHeight(node);
        
        // Get balance factor
        int balance = getBalance(node);
        
        // Perform rotations if needed
        
        // Left Left Case
        if (balance > 1 && getBalance(node.left) >= 0) {
            return rotateRight(node);
        }
        
        // Left Right Case
        if (balance > 1 && getBalance(node.left) < 0) {
            node.left = rotateLeft(node.left);
            return rotateRight(node);
        }
        
        // Right Right Case
        if (balance < -1 && getBalance(node.right) <= 0) {
            return rotateLeft(node);
        }
        
        // Right Left Case
        if (balance < -1 && getBalance(node.right) > 0) {
            node.right = rotateRight(node.right);
            return rotateLeft(node);
        }
        
        return node;
    }
    
    /**
     * Find the minimum node in a subtree
     * @param node Root of the subtree
     * @return Node with minimum value
     */
    private Node findMin(Node node) {
        while (node.left != null) {
            node = node.left;
        }
        return node;
    }
    
    /**
     * Search for a value in the tree
     * @param data The value to search for
     * @return true if found, false otherwise
     */
    public boolean search(T data) {
        return searchHelper(root, data);
    }
    
    /**
     * Recursive helper method for searching
     * @param node Current node in traversal
     * @param data Data to search for
     * @return true if found, false otherwise
     */
    private boolean searchHelper(Node node, T data) {
        if (node == null) {
            return false;
        }
        
        int cmp = data.compareTo(node.data);
        if (cmp == 0) {
            return true;
        } else if (cmp < 0) {
            return searchHelper(node.left, data);
        } else {
            return searchHelper(node.right, data);
        }
    }
    
    /**
     * Get the size of the tree
     * @return Number of elements in the tree
     */
    public int size() {
        return size;
    }
    
    /**
     * Check if the tree is empty
     * @return true if empty, false otherwise
     */
    public boolean isEmpty() {
        return size == 0;
    }
    
    /**
     * Get the height of the tree
     * @return Height of the tree
     */
    public int getTreeHeight() {
        return getHeight(root);
    }
    
    /**
     * Perform inorder traversal
     * @return String representation of inorder traversal
     */
    public String inorderTraversal() {
        StringBuilder sb = new StringBuilder();
        inorderHelper(root, sb);
        return sb.toString().trim();
    }
    
    /**
     * Recursive helper for inorder traversal
     * @param node Current node
     * @param sb StringBuilder to append to
     */
    private void inorderHelper(Node node, StringBuilder sb) {
        if (node != null) {
            inorderHelper(node.left, sb);
            sb.append(node.data).append(" ");
            inorderHelper(node.right, sb);
        }
    }
    
    /**
     * Validate that the tree maintains AVL property
     * @return true if valid AVL tree, false otherwise
     */
    public boolean isValidAVL() {
        return isValidAVLHelper(root);
    }
    
    /**
     * Recursive helper to validate AVL property
     * @param node Current node
     * @return true if subtree is valid AVL, false otherwise
     */
    private boolean isValidAVLHelper(Node node) {
        if (node == null) {
            return true;
        }
        
        int balance = getBalance(node);
        if (Math.abs(balance) > 1) {
            return false;
        }
        
        return isValidAVLHelper(node.left) && isValidAVLHelper(node.right);
    }
    
    /**
     * Demo method showing AVL tree usage
     */
    public static void main(String[] args) {
        AVLTree<Integer> avl = new AVLTree<>();
        
        // Insert values
        int[] values = {10, 20, 30, 40, 50, 25};
        System.out.println("Inserting values: " + java.util.Arrays.toString(values));
        
        for (int value : values) {
            avl.insert(value);
            System.out.println("After inserting " + value + ": " + avl.inorderTraversal());
            System.out.println("Tree height: " + avl.getTreeHeight());
            System.out.println("Is valid AVL: " + avl.isValidAVL());
            System.out.println();
        }
        
        // Search operations
        System.out.println("Search 25: " + avl.search(25));
        System.out.println("Search 35: " + avl.search(35));
        
        // Delete operations
        System.out.println("\nDeleting 30...");
        avl.delete(30);
        System.out.println("After deletion: " + avl.inorderTraversal());
        System.out.println("Tree height: " + avl.getTreeHeight());
        System.out.println("Is valid AVL: " + avl.isValidAVL());
    }
}