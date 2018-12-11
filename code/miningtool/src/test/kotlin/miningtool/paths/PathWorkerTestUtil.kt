package miningtool.paths

import miningtool.common.ASTPath
import miningtool.common.Node
import miningtool.common.postOrder
import miningtool.impl.antlr.SimpleNode
import org.junit.Assert

fun simpleNode(number: Int, parent: Node?): SimpleNode {
    return SimpleNode("$number", parent, "node_$number")
}

fun simpleNodes(numbers: List<Int>, parent: Node?): List<SimpleNode> {
    return numbers.map { simpleNode(it, parent) }
}

fun getPathsCountWithNoHeightLimit(leavesCount: Int, maxWidth: Int): Int {
    if (maxWidth >= leavesCount) return (leavesCount * (leavesCount - 1)) / 2
    return (leavesCount - maxWidth) * maxWidth + (maxWidth * (maxWidth - 1)) / 2
}

fun countPossiblePaths(rootNode: Node, maxHeight: Int, maxWidth: Int): Int {
    val allLeaves = rootNode.postOrder().filter { it.isLeaf() }
    val leaveOrders = allLeaves.mapIndexed { index, node -> Pair(node, index) }.toMap()

    fun Node.retrieveParentsUpToMaxHeight(maxHeight: Int): List<Node> {
        val parents: MutableList<Node> = ArrayList()
        var currentNode = this.getParent()
        while (currentNode != null && parents.size < maxHeight) {
            parents.add(currentNode)
            currentNode = currentNode.getParent()
        }
        return parents
    }

    fun Node.countPathsInSubtreeStartingFrom(startNode: Node, maxWidth: Int): Int {
        val branchIndices : MutableMap<Node, Int>  = HashMap()
        this.getChildren().forEachIndexed { index, node ->
            val childSubTreeLeaves = node.postOrder().filter { it.isLeaf() }
            childSubTreeLeaves.forEach { branchIndices[it] = index }
        }

        val startNodeOrder = leaveOrders[startNode]!!
        val startNodeBranchIndex = branchIndices[startNode]!!

        val possibleEndNodes = this.postOrder().filter {
            it.isLeaf()
                    && (branchIndices[it]!! > startNodeBranchIndex)
                    && (leaveOrders[it]!! > startNodeOrder)
                    && (leaveOrders[it]!! - startNodeOrder) <= maxWidth
        }
        return possibleEndNodes.size
    }

    var totalPaths = 0

    allLeaves.forEach { leaf ->
        val possibleTopNodes = leaf.retrieveParentsUpToMaxHeight(maxHeight)
        possibleTopNodes.forEach { topNode ->
            totalPaths += topNode.countPathsInSubtreeStartingFrom(leaf, maxWidth)
        }
    }

    return totalPaths
}

fun ASTPath.allNodesAreDistinct(): Boolean {
    return this.upwardNodes.size == HashSet(this.upwardNodes).size
            && this.downwardNodes.size == HashSet(this.downwardNodes).size
}

fun ASTPath.piecesMatch(): Boolean = this.upwardNodes.last() === this.downwardNodes.first()

fun assertPathIsValid(path: ASTPath) {
    Assert.assertTrue(path.allNodesAreDistinct())
    Assert.assertTrue(path.piecesMatch())
}