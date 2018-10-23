package codestyle.miner

import com.github.gumtreediff.tree.ITree

val MIN_HEIGHT_KEY = "minHeight"
val MAX_HEIGHT_KEY = "maxHeight"
val MIN_LEAF_INDEX_KEY = "minLeafIndex"
val MAX_LEAF_INDEX_KEY = "maxLeafIndex"
val PATH_PIECES_KEY = "pathPieces"

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

fun ITree.setMinHeight(value: Int) = setIntValue(MIN_HEIGHT_KEY, value)
fun ITree.setMaxHeight(value: Int) = setIntValue(MAX_HEIGHT_KEY, value)
fun ITree.setMinLeafIndex(value: Int) = setIntValue(MIN_LEAF_INDEX_KEY, value)
fun ITree.setMaxLeafIndex(value: Int) = setIntValue(MAX_LEAF_INDEX_KEY, value)

fun ITree.getMinHeight() = getIntValue(MIN_HEIGHT_KEY)
fun ITree.getMaxHeight() = getIntValue(MAX_HEIGHT_KEY)
fun ITree.getMinLeafIndex() = getIntValue(MIN_LEAF_INDEX_KEY)
fun ITree.getMaxLeafIndex() = getIntValue(MAX_LEAF_INDEX_KEY)

class Path(val firstPiece: List<ITree>, val secondPiece: List<ITree>){

}



fun traverseTree(root: ITree) {
    val iterator = root.postOrder()
    var currentLeafIndex = 0
    iterator.forEach {
        if (it.isLeaf) {
            val leafIndex = currentLeafIndex++
            it.setMinLeafIndex(leafIndex)
            it.setMaxLeafIndex(leafIndex)
            it.setMinHeight(0)
            it.setMaxHeight(0)
            it.setPathPieces(listOf(listOf(it)))
        } else {
            it.setMinLeafIndex(it.children.map { child -> child.getMinLeafIndex() }.min() ?: -1)
            it.setMaxLeafIndex(it.children.map { child -> child.getMaxLeafIndex() }.max() ?: -1)
            it.setMinHeight((it.children.map { child -> child.getMinHeight() }.min() ?: -2) + 1)
            it.setMaxHeight((it.children.map { child -> child.getMaxHeight() }.max() ?: -2) + 1)

            val childPathPieces = it.children.map { it.getPathPieces() }.flatten()
            val currentNodePathPieces = childPathPieces.map { l -> l + it }


        }


    }


}