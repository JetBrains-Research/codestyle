package codestyle.miner

import com.github.gumtreediff.tree.ITree
import com.github.gumtreediff.tree.TreeContext

const val LEAF_INDEX_KEY = "leafIndex"
const val PATH_PIECES_KEY = "pathPieces"

fun ITree.setIntValue(key: String, value: Int) {
    this.setMetadata(key, value)
}

fun ITree.getIntValue(key: String): Int {
    return this.getMetadata(key) as Int
}

fun ITree.setPathPieces(pathPieces: Collection<List<ITree>>) {
    this.setMetadata(PATH_PIECES_KEY, pathPieces)
}

fun ITree.getPathPieces(): Collection<List<ITree>> = this.getMetadata(PATH_PIECES_KEY) as Collection<List<ITree>>

fun ITree.setLeafIndex(value: Int) = setIntValue(LEAF_INDEX_KEY, value)

fun ITree.getMinLeafIndex() = getIntValue(LEAF_INDEX_KEY)


fun getPathsForCurrentNode(pathPieces: Collection<List<ITree>>,
                           maxLength: Int, maxWidth: Int,
                           treeContext: TreeContext,
                           pathStorage: PathStorage,
                           matchedPieces: MutableSet<Pair<List<ITree>, List<ITree>>>): Collection<PathContext> {
    val paths: MutableCollection<PathContext> = ArrayList()
    val sortedPieces = pathPieces.sortedBy { (it[0].getMinLeafIndex()) }
    sortedPieces.forEachIndexed { index, upPiece ->
        for (i in (index + 1 until sortedPieces.size)) {
            val downPiece = sortedPieces[i]
            val length = upPiece.size + downPiece.size - 1 // -1 as the top node is present in both pieces
            val width = downPiece[0].getMinLeafIndex() - upPiece[0].getMinLeafIndex()
            val piecePair = Pair(upPiece, downPiece)
            if (length <= maxLength && width <= maxWidth && piecePair !in matchedPieces) {
                paths.add(pathStorage.store(upPiece, downPiece, treeContext))
                matchedPieces.add(piecePair)
            }
        }
    }
    return paths
}

fun retrievePaths(treeContext: TreeContext, startNode: ITree, pathStorage: PathStorage) = retrievePaths(treeContext, startNode, pathStorage, Int.MAX_VALUE, Int.MAX_VALUE)

fun retrievePaths(treeContext: TreeContext, startNode: ITree, pathStorage: PathStorage, maxLength: Int, maxWidth: Int): Collection<PathContext> {
    val iterator = startNode.postOrder()
    var currentLeafIndex = 0
    val paths: MutableCollection<PathContext> = ArrayList()
    val matchedPieces: MutableSet<Pair<List<ITree>, List<ITree>>> = HashSet()
    iterator.forEach {
        if (it.isLeaf) {
            val leafIndex = currentLeafIndex++
            it.setLeafIndex(leafIndex)
            it.setPathPieces(listOf(listOf(it)))
        } else {

            val childPathPieces = it.children.map { it.getPathPieces() }.flatten()

            val currentNodePathPieces = childPathPieces
                    // Filtering out the paths that are already too long.
                    // -2 represent the current node and its possible immediate leaf child.
                    .filter { pathPiece -> pathPiece.size <= maxLength - 2 }
                    // Appending the current node to every piece
                    .map { l -> l + it }

            it.setPathPieces(currentNodePathPieces)
            paths.addAll(getPathsForCurrentNode(currentNodePathPieces, maxLength, maxWidth, treeContext, pathStorage, matchedPieces))
        }
    }
    return paths
}