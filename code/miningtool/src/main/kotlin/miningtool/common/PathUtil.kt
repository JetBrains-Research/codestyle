package miningtool.common

fun ASTPath.toPathContext(): PathContext {
    val startToken = this.upwardNodes.first().getToken()
    val endToken = this.downwardNodes.last().getToken()
    val path = this.upwardNodes
            .takeLast(this.upwardNodes.size - 1).map { NodeType(it.getTypeLabel(), Direction.UP) } +
            this.downwardNodes.take(this.downwardNodes.size - 1).map { NodeType(it.getTypeLabel(), Direction.DOWN) }
    return PathContext(startToken, path, endToken)
}