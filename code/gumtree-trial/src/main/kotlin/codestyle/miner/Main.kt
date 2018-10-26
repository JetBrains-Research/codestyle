package codestyle.miner

import com.github.gumtreediff.client.Run
import com.github.gumtreediff.tree.ITree
import com.github.gumtreediff.tree.TreeContext
import com.google.common.io.Files
import com.google.gson.GsonBuilder
import java.io.File
import java.io.FileWriter
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

private fun parseChangeEntry(id: Int, csvLine: String, csvSettings: CsvSettings): ChangeEntry {
    val values = csvLine.split(',')

    return ChangeEntry(
            id,
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


fun processEntries(entries: List<ChangeEntry>, pathStorage: PathStorage): MutableList<FileChangeInfo> {
    val nCores = Runtime.getRuntime().availableProcessors()
    val nchunks = nCores - 1
    val chunkSize = (entries.size / nchunks) + 1
    println("have $nCores cores, running $nchunks threads processing $chunkSize entries each")

    val threads: MutableCollection<Thread> = HashSet()
    val infos: MutableList<FileChangeInfo> = ArrayList()

    entries.chunked(chunkSize).forEach { chunk ->
        val currentThread = thread {
            chunk.forEach {
                val info = processChangeEntry(it, pathStorage)
                synchronized(pathStorage) {
                    infos.add(info)
                }
            }
        }
        threads.add(currentThread)
    }

    threads.forEach {
        it.join()
    }

    return infos
}

fun processRepositoryData() {
    val blobListFile = "../python-miner/data/exploded/intellij-community/infos_full.csv"
    val lines = Files.readLines(File(blobListFile), Charsets.UTF_8)
    val settings = CsvSettings(lines.first())
    println("${lines.size} entries read")

    Run.initGenerators()

    val startTime = System.currentTimeMillis()

    val pathStorage = PathStorage()

    var counter = 0
    fun getId(): Int = counter++

    val entries = lines.drop(1).take(100_000).map { parseChangeEntry(getId(), it, settings) }

    val infos = processEntries(entries, pathStorage)

    dumpData(entries, infos, pathStorage)


    val elapsed = System.currentTimeMillis() - startTime
    println("Processed ${entries.size} entries in ${elapsed / 1000} seconds (${1000.0 * entries.size / elapsed} entries/s)")
}

fun getMappingContext(entry: ChangeEntry): MappingContext {
    return getMappingContext(entry.oldContentId, entry.newContentId)
}

fun PathContext.toShortString(): String = "${this.startToken} ${this.pathId} ${this.endToken}"

fun processChangeEntry(entry: ChangeEntry, pathStorage: PathStorage): FileChangeInfo {
    // retrieve the method mappings between the two versions of the file
    val mappingContext = getMappingContext(entry)

    // extract the changed methods
    val changedMappings = mappingContext.mappings.filter { it.isChanged }

    fun getMethodPaths(node: ITree?, context: TreeContext?): Collection<PathContext> {
        if (node == null || context == null) return emptyList()
        return retrievePaths(context, node, pathStorage, 10, 3)
    }


    val methodChangeInfos: MutableList<MethodChangeInfo> = ArrayList()

    changedMappings.forEach {
        val treeBefore = it.before?.node
        val treeAfter = it.after?.node

        val pathsBefore = getMethodPaths(treeBefore, mappingContext.treeContextBefore)
        val pathsAfter = getMethodPaths(treeAfter, mappingContext.treeContextAfter)
        val methodChangeData = MethodChangeInfo(it.before?.id, it.after?.id,
                pathsBefore.size,
                pathsAfter.size,
                pathsBefore.map { path -> path.toShortString() }.joinToString(separator = ";"),
                pathsAfter.map { path -> path.toShortString() }.joinToString(separator = ";"))
        methodChangeInfos.add(methodChangeData)
    }

    return FileChangeInfo(entry.id, methodChangeInfos)
}

fun saveInfosToJson(filename: String, infos: List<FileChangeInfo>) {
        GsonBuilder().setPrettyPrinting().create().toJson(infos, FileWriter(filename))
}
