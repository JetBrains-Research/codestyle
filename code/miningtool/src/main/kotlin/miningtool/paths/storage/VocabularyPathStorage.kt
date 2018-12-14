package miningtool.paths.storage

import miningtool.common.NodeType
import miningtool.common.PathContext
import miningtool.common.PathStorage
import java.io.File

class VocabularyPathStorage : PathStorage() {
    private val tokensMap: IncrementalIdStorage<String> = IncrementalIdStorage()
    private val nodeTypesMap: IncrementalIdStorage<NodeType> = IncrementalIdStorage()
    private val pathsMap: IncrementalIdStorage<List<Long>> = IncrementalIdStorage()

    private val pathContextsPerEntity: MutableMap<String, Collection<PathContextId>> = HashMap()

    data class PathContextId(val startTokenId: Long, val pathId: Long, val endTokenId: Long)

    private fun doStore(pathContext: PathContext): PathContextId {
        val startTokenId = tokensMap.record(pathContext.startToken)
        val endTokenId = tokensMap.record(pathContext.endToken)
        val nodeTypesIds = pathContext.nodeTypes.map { nodeTypesMap.record(it) }
        val pathId = pathsMap.record(nodeTypesIds)
        return PathContextId(startTokenId, pathId, endTokenId)
    }

    override fun store(pathContexts: Collection<PathContext>, entityId: String) {
        val pathContextIds = pathContexts.map { doStore(it) }
        pathContextsPerEntity[entityId] = pathContextIds
    }

    private val nodeTypeToCsvString: (NodeType) -> String = { nt -> "${nt.typeLabel} ${nt.direction}" }
    private val listOfLongToCsvString: (List<Long>) -> String = { list -> list.joinToString(separator = " ") }


    private fun dumpTokenStorage(tokenStorage: IncrementalIdStorage<String>, file: File) {
        dumpIdStorage(tokenStorage, "token", { token -> token }, file)
    }

    private fun dumpNodeTypesStorage(nodeTypesStorage: IncrementalIdStorage<NodeType>, file: File) {
        dumpIdStorage(nodeTypesStorage, "node_type", nodeTypeToCsvString, file)
    }

    private fun dumpPathsStorage(pathsStorage: IncrementalIdStorage<List<Long>>, file: File) {
        dumpIdStorage(pathsMap, "path", listOfLongToCsvString, file)
    }

    private fun savePathContexts(file: File) {
        val lines = mutableListOf("id,path_contexts")
        pathContextsPerEntity.forEach { id, pathContexts ->
            val pathContextsString = pathContexts
                    .map { "${it.startTokenId} ${it.pathId} ${it.endTokenId}" }
                    .joinToString(separator = ";")
            lines.add("$id,$pathContextsString")
        }

        writeLinesToFile(lines, file)
    }

    override fun save(directoryPath: String) {
        File(directoryPath).mkdirs()
        dumpTokenStorage(tokensMap, File("$directoryPath/tokens.csv"))
        dumpNodeTypesStorage(nodeTypesMap, File("$directoryPath/node_types.csv"))
        dumpPathsStorage(pathsMap, File("$directoryPath/paths.csv"))

        savePathContexts(File("$directoryPath/path_contexts.csv"))
    }
}