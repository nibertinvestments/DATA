// Advanced Tree Data Structures in Kotlin
// Comprehensive tree implementations

// AVL Tree (Self-Balancing Binary Search Tree)
class AVLTree<T : Comparable<T>> {
    private data class Node<T>(
        var value: T,
        var height: Int = 1,
        var left: Node<T>? = null,
        var right: Node<T>? = null
    )
    
    private var root: Node<T>? = null
    
    private fun height(node: Node<T>?): Int = node?.height ?: 0
    
    private fun updateHeight(node: Node<T>) {
        node.height = 1 + maxOf(height(node.left), height(node.right))
    }
    
    private fun balanceFactor(node: Node<T>): Int {
        return height(node.left) - height(node.right)
    }
    
    private fun rotateRight(y: Node<T>): Node<T> {
        val x = y.left!!
        val T2 = x.right
        
        x.right = y
        y.left = T2
        
        updateHeight(y)
        updateHeight(x)
        
        return x
    }
    
    private fun rotateLeft(x: Node<T>): Node<T> {
        val y = x.right!!
        val T2 = y.left
        
        y.left = x
        x.right = T2
        
        updateHeight(x)
        updateHeight(y)
        
        return y
    }
    
    private fun balance(node: Node<T>): Node<T> {
        updateHeight(node)
        val balance = balanceFactor(node)
        
        // Left heavy
        if (balance > 1) {
            if (balanceFactor(node.left!!) < 0) {
                node.left = rotateLeft(node.left!!)
            }
            return rotateRight(node)
        }
        
        // Right heavy
        if (balance < -1) {
            if (balanceFactor(node.right!!) > 0) {
                node.right = rotateRight(node.right!!)
            }
            return rotateLeft(node)
        }
        
        return node
    }
    
    fun insert(value: T) {
        root = insertHelper(root, value)
    }
    
    private fun insertHelper(node: Node<T>?, value: T): Node<T> {
        if (node == null) return Node(value)
        
        when {
            value < node.value -> node.left = insertHelper(node.left, value)
            value > node.value -> node.right = insertHelper(node.right, value)
            else -> return node
        }
        
        return balance(node)
    }
    
    fun search(value: T): Boolean {
        var current = root
        while (current != null) {
            when {
                value < current.value -> current = current.left
                value > current.value -> current = current.right
                else -> return true
            }
        }
        return false
    }
    
    fun inorderTraversal(): List<T> {
        val result = mutableListOf<T>()
        inorderHelper(root, result)
        return result
    }
    
    private fun inorderHelper(node: Node<T>?, result: MutableList<T>) {
        if (node != null) {
            inorderHelper(node.left, result)
            result.add(node.value)
            inorderHelper(node.right, result)
        }
    }
}

// Red-Black Tree
class RedBlackTree<T : Comparable<T>> {
    private enum class Color { RED, BLACK }
    
    private data class Node<T>(
        var value: T,
        var color: Color = Color.RED,
        var left: Node<T>? = null,
        var right: Node<T>? = null,
        var parent: Node<T>? = null
    )
    
    private var root: Node<T>? = null
    
    fun insert(value: T) {
        val newNode = Node(value)
        
        if (root == null) {
            root = newNode
            newNode.color = Color.BLACK
            return
        }
        
        // Standard BST insert
        var current = root
        var parent: Node<T>? = null
        
        while (current != null) {
            parent = current
            current = if (value < current.value) current.left else current.right
        }
        
        newNode.parent = parent
        if (value < parent!!.value) {
            parent.left = newNode
        } else {
            parent.right = newNode
        }
        
        // Fix Red-Black tree properties
        fixInsert(newNode)
    }
    
    private fun fixInsert(node: Node<T>) {
        var current = node
        
        while (current != root && current.parent?.color == Color.RED) {
            val parent = current.parent!!
            val grandparent = parent.parent!!
            
            if (parent == grandparent.left) {
                val uncle = grandparent.right
                
                if (uncle?.color == Color.RED) {
                    parent.color = Color.BLACK
                    uncle.color = Color.BLACK
                    grandparent.color = Color.RED
                    current = grandparent
                } else {
                    if (current == parent.right) {
                        current = parent
                        rotateLeft(current)
                    }
                    parent.color = Color.BLACK
                    grandparent.color = Color.RED
                    rotateRight(grandparent)
                }
            } else {
                val uncle = grandparent.left
                
                if (uncle?.color == Color.RED) {
                    parent.color = Color.BLACK
                    uncle.color = Color.BLACK
                    grandparent.color = Color.RED
                    current = grandparent
                } else {
                    if (current == parent.left) {
                        current = parent
                        rotateRight(current)
                    }
                    parent.color = Color.BLACK
                    grandparent.color = Color.RED
                    rotateLeft(grandparent)
                }
            }
        }
        
        root?.color = Color.BLACK
    }
    
    private fun rotateLeft(node: Node<T>) {
        val rightChild = node.right!!
        node.right = rightChild.left
        
        if (rightChild.left != null) {
            rightChild.left!!.parent = node
        }
        
        rightChild.parent = node.parent
        
        when {
            node.parent == null -> root = rightChild
            node == node.parent!!.left -> node.parent!!.left = rightChild
            else -> node.parent!!.right = rightChild
        }
        
        rightChild.left = node
        node.parent = rightChild
    }
    
    private fun rotateRight(node: Node<T>) {
        val leftChild = node.left!!
        node.left = leftChild.right
        
        if (leftChild.right != null) {
            leftChild.right!!.parent = node
        }
        
        leftChild.parent = node.parent
        
        when {
            node.parent == null -> root = leftChild
            node == node.parent!!.right -> node.parent!!.right = leftChild
            else -> node.parent!!.left = leftChild
        }
        
        leftChild.right = node
        node.parent = leftChild
    }
    
    fun search(value: T): Boolean {
        var current = root
        while (current != null) {
            when {
                value < current.value -> current = current.left
                value > current.value -> current = current.right
                else -> return true
            }
        }
        return false
    }
}

// B-Tree
class BTree<T : Comparable<T>>(private val degree: Int = 3) {
    private data class Node<T>(
        val keys: MutableList<T> = mutableListOf(),
        val children: MutableList<Node<T>> = mutableListOf(),
        var isLeaf: Boolean = true
    )
    
    private var root = Node<T>()
    
    fun insert(key: T) {
        if (root.keys.size == 2 * degree - 1) {
            val newRoot = Node<T>()
            newRoot.isLeaf = false
            newRoot.children.add(root)
            splitChild(newRoot, 0)
            root = newRoot
        }
        insertNonFull(root, key)
    }
    
    private fun insertNonFull(node: Node<T>, key: T) {
        var i = node.keys.size - 1
        
        if (node.isLeaf) {
            node.keys.add(key)
            while (i >= 0 && key < node.keys[i]) {
                node.keys[i + 1] = node.keys[i]
                i--
            }
            node.keys[i + 1] = key
        } else {
            while (i >= 0 && key < node.keys[i]) {
                i--
            }
            i++
            
            if (node.children[i].keys.size == 2 * degree - 1) {
                splitChild(node, i)
                if (key > node.keys[i]) {
                    i++
                }
            }
            insertNonFull(node.children[i], key)
        }
    }
    
    private fun splitChild(parent: Node<T>, index: Int) {
        val fullChild = parent.children[index]
        val newChild = Node<T>()
        newChild.isLeaf = fullChild.isLeaf
        
        val midIndex = degree - 1
        
        // Move half of keys to new node
        for (j in 0 until degree - 1) {
            newChild.keys.add(fullChild.keys[midIndex + 1 + j])
        }
        
        // Move children if not leaf
        if (!fullChild.isLeaf) {
            for (j in 0 until degree) {
                newChild.children.add(fullChild.children[midIndex + 1 + j])
            }
        }
        
        // Update parent
        parent.keys.add(index, fullChild.keys[midIndex])
        parent.children.add(index + 1, newChild)
        
        // Remove moved keys from full child
        fullChild.keys.subList(midIndex, fullChild.keys.size).clear()
        if (!fullChild.isLeaf) {
            fullChild.children.subList(midIndex + 1, fullChild.children.size).clear()
        }
    }
    
    fun search(key: T): Boolean {
        return searchHelper(root, key)
    }
    
    private fun searchHelper(node: Node<T>, key: T): Boolean {
        var i = 0
        while (i < node.keys.size && key > node.keys[i]) {
            i++
        }
        
        if (i < node.keys.size && key == node.keys[i]) {
            return true
        }
        
        return if (node.isLeaf) {
            false
        } else {
            searchHelper(node.children[i], key)
        }
    }
}

fun main() {
    println("Advanced Tree Data Structures in Kotlin")
    println("========================================")
    
    // Test AVL Tree
    val avl = AVLTree<Int>()
    listOf(10, 20, 30, 40, 50, 25).forEach { avl.insert(it) }
    println("AVL Tree inorder: ${avl.inorderTraversal()}")
    println("Search 25: ${avl.search(25)}")
}
