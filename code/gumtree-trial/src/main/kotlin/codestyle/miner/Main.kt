package codestyle.miner

import com.github.gumtreediff.client.Run
import com.github.gumtreediff.tree.ITree
import com.github.gumtreediff.tree.TreeContext
import com.google.common.io.Files
import java.io.File
import kotlin.concurrent.thread

fun main(args: Array<String>) {
    processRepositoryData()
}

class CsvSettings(csvHeader: String) {
    private val keyPositions: MutableMap<String, Int> = HashMap()

    init {
        csvHeader.split(',').forEachIndexed { index, key ->
            keyPositions[key] = index
        }
    }

    fun getKeyIndex(key: String): Int {
        return keyPositions[key] ?: -1
    }
}

private fun nullIfEmpty(s: String) = if (s.isEmpty()) null else s

private fun createBlobId(idString: String): BlobId? {
    if (idString.isEmpty()) return null
    return BlobId(idString)
}

private fun parseChangeEntry(csvLine: String, csvSettings: CsvSettings): ChangeEntry {
    val values = csvLine.split(',')

    return ChangeEntry(
            values[csvSettings.getKeyIndex("commit_id")],
            values[csvSettings.getKeyIndex("author_name")],
            values[csvSettings.getKeyIndex("author_email")],
            values[csvSettings.getKeyIndex("committer_name")],
            values[csvSettings.getKeyIndex("committer_email")],
            values[csvSettings.getKeyIndex("author_time")].toLong(),
            values[csvSettings.getKeyIndex("committer_time")].toLong(),
            values[csvSettings.getKeyIndex("change_type")].first(),
            createBlobId(values[csvSettings.getKeyIndex("old_content")]),
            createBlobId(values[csvSettings.getKeyIndex("new_content")]),
            nullIfEmpty(values[csvSettings.getKeyIndex("old_path")]),
            nullIfEmpty(values[csvSettings.getKeyIndex("new_path")])
    )
}


fun processEntries(entries: List<ChangeEntry>, pathStorage: PathStorage) {
    val nCores = Runtime.getRuntime().availableProcessors()
    val nchunks = nCores - 1
    val chunkSize = (entries.size / nchunks) + 1
    println("have $nCores cores, running $nchunks threads processing $chunkSize entries each")

    val threads: MutableCollection<Thread> = HashSet()

    entries.chunked(chunkSize).forEach { chunk ->
        val currentThread = thread {
            chunk.forEach {
                processChangeEntry(it, pathStorage)
            }
        }
        threads.add(currentThread)
    }

    threads.forEach {
        it.join()
    }
}

fun processRepositoryData(): List<String> {
    val blobListFile = "../python-miner/data/exploded/intellij-community/infos_full.csv"
    val lines = Files.readLines(File(blobListFile), Charsets.UTF_8)
    val settings = CsvSettings(lines.first())
    println("${lines.size} entries read")

    Run.initGenerators()

    val startTime = System.currentTimeMillis()

    val pathStorage = PathStorage()

    val entries = lines.drop(1).map { parseChangeEntry(it, settings) }

    processEntries(entries, pathStorage)

    val elapsed = System.currentTimeMillis() - startTime
    println("Processed ${lines.size} entries in ${elapsed / 1000} seconds (${1000.0 * lines.size / elapsed} entries/s)")
    return lines
}

fun getMappingContext(entry: ChangeEntry): MappingContext {
    return getMappingContext(entry.oldContentId, entry.newContentId)
}

fun processChangeEntry(entry: ChangeEntry, pathStorage: PathStorage): Collection<Path> {
    // retrieve the method mappings between the two versions of the file
    val mappingContext = getMappingContext(entry)

    if (mappingContext.treeContextBefore == null || mappingContext.treeContextAfter == null) {
        //todo handle
        return emptyList()
    }

    // extract the changed methods
    val changedMappings = mappingContext.mappings.filter { it.isChanged }

    fun getMethodPaths(node: ITree?, context: TreeContext): Collection<PathContext> {
        if(node == null) return emptyList()
        return retrievePaths(context, node, pathStorage, 10, 3)
    }

    changedMappings.forEach {
        val treeBefore = it.before?.node
        val treeAfter = it.after?.node

        val pathsBefore = getMethodPaths(treeBefore, mappingContext.treeContextBefore)
        val pathsAfter = getMethodPaths(treeAfter, mappingContext.treeContextAfter)

//        println("Before: ${pathsBefore.size} paths, after: ${pathsAfter.size}")
    }

    return emptyList()

}
