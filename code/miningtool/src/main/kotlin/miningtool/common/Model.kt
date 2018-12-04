package miningtool.common

import java.io.InputStream

interface Node {
    fun getTypeLabel(): String
    fun getChildren(): List<Node>
    fun getParent(): Node?
    fun getToken(): String
    fun isLeaf(): Boolean
}


//class SimpleNode(val typeLabel: String, val parent: Node?, val token: String): AbstractNode() {}

interface TreeSplitter<T: Node> {
    fun split(root: T): Collection<T>
}

data class Path(val upwardNodes: List<Node>, val downwardNodes: List<Node>)
data class PathContext(val startToken: String, val path: Path, val endToken: String)

abstract class Parser {
    abstract fun parse(content: InputStream): Node?
}
