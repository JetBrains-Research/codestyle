package miningtool.impl.java

import com.github.gumtreediff.tree.ITree
import com.github.gumtreediff.tree.TreeContext
import miningtool.common.Node

class GumTreeJavaNode(val wrappedNode: ITree, val context: TreeContext, val parent: GumTreeJavaNode?) : Node {
    val myMetadata: MutableMap<String, Any> = HashMap()

    override fun getMetadata(key: String): Any? {
        return myMetadata[key]
    }

    override fun setMetadata(key: String, value: Any) {
        myMetadata[key] = value
    }

    override fun isLeaf(): Boolean {
        return childrenList.isEmpty()
    }

    private val childrenList: List<GumTreeJavaNode> by lazy {
        wrappedNode.children.map { GumTreeJavaNode(it, context, this) }
    }

    override fun getTypeLabel(): String {
        return context.getTypeLabel(wrappedNode)
    }

    override fun getChildren(): List<Node> {
        return childrenList
    }

    override fun getParent(): Node? {
        return parent
    }

    override fun getToken(): String {
        return wrappedNode.label
    }
}