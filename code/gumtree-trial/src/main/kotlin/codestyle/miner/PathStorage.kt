package codestyle.miner

import com.github.gumtreediff.tree.ITree
import com.github.gumtreediff.tree.TreeContext


data class PathContext(val startToken: Long, val pathId: Long, val endToken: Long)
data class Path(val startToken: String, val upwardNodeTypes: List<String>, val downwardNodeTypes: List<String>, val endToken: String)

enum class Direction { UP, DOWN }

data class NodeType(val direction: Direction, val type: String)

class IncrementalIdStorage<T> {
    private var recordCounter = 0L
    private var keyCounter = 0L
    val map: MutableMap<T, Long> = HashMap()

    private fun putAndIncrement(item: T): Long {
        map[item] = ++keyCounter
        return keyCounter
    }

    fun record(item: T): Long {
        recordCounter++
        return map[item] ?: putAndIncrement(item)
    }

    fun get(id: Long): T? {
        return map.entries.first{it.value == id}.key
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
        val start = tokenIds.get(startToken)
        val end = tokenIds.get(endToken)
        val path = pathIds.get(pathId)?.map { nodeTypeIds.get(it) }
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