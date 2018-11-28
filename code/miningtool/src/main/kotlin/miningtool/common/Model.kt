package miningtool.common

import java.io.InputStream

abstract class Node {
    abstract fun getTypeLabel(): String
    abstract fun getChildren(): List<Node>
    abstract fun getParent(): Node?
    abstract fun getToken(): String
    override fun toString(): String {
        return "${getTypeLabel()} ${getToken()}"
    }

    fun isLeaf(): Boolean = getChildren().isEmpty()
}

interface TreeSplitter<T: Node> {
    fun split(root: T): Collection<T>
}

data class Path(val upwardNodes: List<Node>, val downwardNodes: List<Node>)
data class PathContext(val startToken: String, val path: Path, val endToken: String)

abstract class Parser {
    abstract fun parse(content: InputStream): Node?
}
