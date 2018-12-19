package miningtool.examples

import miningtool.common.*
import miningtool.parse.antlr.java.JavaParser
import miningtool.paths.PathMiner
import miningtool.paths.PathRetrievalSettings
import miningtool.paths.storage.VocabularyPathStorage
import java.io.File


fun allJavaFiles() {
    val folder = "./testData"

    val miner = PathMiner(PathRetrievalSettings(5, 5))

    val storage = VocabularyPathStorage()

    File(folder).walkTopDown().filter { it.path.endsWith(".java") }.forEach { file ->
        val node = JavaParser().parse(file.inputStream()) ?: return@forEach
        val paths = miner.retrievePaths(node)

        println(file.path)
        println(paths.size)

        storage.store(paths.map { it.toPathContext() }, entityId = file.path)
    }

    storage.save("examples/out/allJavaFiles")
}