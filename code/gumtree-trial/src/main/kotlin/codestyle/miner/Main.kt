package codestyle.miner

import com.google.common.io.Files
import java.io.File

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


fun processChangeEntry(entry: ChangeEntry) {

}


fun processRepositoryData(): List<String> {
    val blobListFile = "../python-miner/data/exploded/intellij-community/infos_full.csv"
    val lines = Files.readLines(File(blobListFile), Charsets.UTF_8)
    val settings = CsvSettings(lines.first())

    lines.drop(1).map { parseChangeEntry(it, settings) }.filter { it.changeType!='M' }.forEach {
        processChangeEntry(it)
    }


    return lines
}
