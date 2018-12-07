package miningtool.common

import java.io.InputStream

interface Node {
    fun getTypeLabel(): String
    fun getChildren(): List<Node>
    fun getParent(): Node?
    fun getToken(): String
    fun isLeaf(): Boolean

    fun getMetadata(key: String): Any?
    fun setMetadata(key: String, value: Any)
}

interface TreeSplitter<T: Node> {
    fun split(root: T): Collection<T>
}

data class ASTPath(val upwardNodes: List<Node>, val downwardNodes: List<Node>)
data class PathContext(val startToken: String, val path: ASTPath, val endToken: String)

abstract class Parser {
    abstract fun parse(content: InputStream): Node?
}
