package codestyle.miner

import com.github.gumtreediff.tree.ITree
import com.github.gumtreediff.tree.TreeContext


data class PathContext(val startToken: Long, val pathId: Long, val endToken: Long)
data class Path(val startToken: String, val upwardNodeTypes: List<String>, val downwardNodeTypes: List<String>, val endToken: String)

enum class Direction { UP, DOWN }

data class NodeType(val direction: Direction, val type: String)

class IncrementalIdStorage<T> {
    private var keyCounter = 0L
    val valueMap: MutableMap<T, Long> = HashMap()
    private val idCountMap: MutableMap<Long, Long> = HashMap()

    private fun putAndIncrementKey(item: T): Long {
        valueMap[item] = ++keyCounter
        return keyCounter
    }

    private fun incrementIdCount(id: Long) {
        val count = idCountMap[id] ?: 0
        idCountMap[id] = count + 1
    }

    fun record(item: T): Long {
        val id = valueMap[item] ?: putAndIncrementKey(item)
        incrementIdCount(id)
        return id
    }

    fun getIdCount(id: Long): Long {
        return idCountMap[id]?:0
    }

    fun getValue(id: Long): T? {
        return valueMap.entries.first { it.value == id }.key
    }
}

fun createPath(upward: List<ITree>, downward: List<ITree>, treeContext: TreeContext): Path {
    val startToken = upward[0].label
    val endToken = downward[0].label

    return Path(startToken,
            upward.map { treeContext.getTypeLabel(it) },
            downward.reversed().map { treeContext.getTypeLabel(it) },
            endToken)
}

class PathStorage {

    val tokenIds: IncrementalIdStorage<String> = IncrementalIdStorage()

    val nodeTypeIds: IncrementalIdStorage<NodeType> = IncrementalIdStorage()
    val pathIds: IncrementalIdStorage<List<Long>> = IncrementalIdStorage()

    private fun storePath(upward: List<String>, downward: List<String>): Long {
        val nodeIds: MutableList<Long> = ArrayList()
        upward.forEach {
            val nodeType = NodeType(Direction.UP, it)
            val id = nodeTypeIds.record(nodeType)
            nodeIds.add(id)
        }
        downward.forEach {
            val nodeType = NodeType(Direction.DOWN, it)
            val id = nodeTypeIds.record(nodeType)
            nodeIds.add(id)
        }
        return pathIds.record(nodeIds)
    }

    fun getPathString(startToken: Long, pathId: Long, endToken: Long): String {
        val start = tokenIds.getValue(startToken)
        val end = tokenIds.getValue(endToken)
        val path = pathIds.getValue(pathId)?.map { nodeTypeIds.getValue(it) }
        return "$start $path $end"
    }

    @Synchronized
    fun store(upward: List<ITree>, downward: List<ITree>, context: TreeContext): PathContext {
        val path = createPath(upward, downward, context)
        val pathId = storePath(path.upwardNodeTypes, path.downwardNodeTypes)
        val startTokenId = tokenIds.record(path.startToken)
        val endTokenId = tokenIds.record(path.endToken)

        return PathContext(startTokenId, pathId, endTokenId)
    }
}