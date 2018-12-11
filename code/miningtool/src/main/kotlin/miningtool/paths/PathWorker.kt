package miningtool.paths

import miningtool.common.ASTPath
import miningtool.common.Node

class PathWorker {
    fun retrievePaths(tree: Node) = retrievePaths(tree, Int.MAX_VALUE, Int.MAX_VALUE)

    fun retrievePaths(tree: Node, maxHeight: Int, maxWidth: Int): Collection<ASTPath> {
        return emptyList()
    }
}