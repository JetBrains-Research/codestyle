package codestyle.miner

import com.github.gumtreediff.client.Run
import com.github.gumtreediff.gen.jdt.JdtTreeGenerator
import com.google.common.io.Files
import java.io.File

fun main(args: Array<String>) {
    println("hello world")
    Run.initGenerators()
    read_blob_filenames()
}

fun readAndParseBlob(id: String) {
    val file = "../python-miner/data/exploded/intellij-community/blobs/$id"
    val treeContext = JdtTreeGenerator().generateFromFile(file)
//    println(treeContext)
}

fun read_blob_filenames(): List<String> {
    val blob_list_file = "../python-miner/data/exploded/intellij-community/parse_status.csv"
    val lines = Files.readLines(File(blob_list_file), Charsets.UTF_8)
    var count = 0
    lines.drop(1).forEach {
        val components = it.split(",")
        val id = components[0]
        val status = components[2]
        if (status == "OK") {
            readAndParseBlob(id)
        }
        count++
        println(count)
    }


    return lines
}
