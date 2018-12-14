package miningtool.examples

import miningtool.common.*
import miningtool.parse.antlr.java.JavaParser
import miningtool.paths.PathMiner
import miningtool.paths.PathRetrievalSettings
import miningtool.paths.storage.VocabularyPathStorage
import java.io.File


fun ASTPath.toPathContext(): PathContext {
    val startToken = this.upwardNodes.first().getToken()
    val endToken = this.downwardNodes.last().getToken()
    val path = this.upwardNodes
            .takeLast(this.upwardNodes.size - 1).map { NodeType(it.getTypeLabel(), Direction.UP) } +
            this.downwardNodes.take(this.downwardNodes.size - 1).map { NodeType(it.getTypeLabel(), Direction.DOWN) }
    return PathContext(startToken, path, endToken)
}

fun allJavaFiles() {
    val folder = "./testData"

    val parserProvider: () -> Parser = { JavaParser() }
    val miner = PathMiner(parserProvider, PathRetrievalSettings(5, 5))

    val storage = VocabularyPathStorage()

    File(folder).walkTopDown().filter { it.path.endsWith(".java") }.forEach { file ->
        val paths = miner.retrievePaths(file.inputStream())



        println(file.path)
        println(paths.size)

        storage.store(paths.map { it.toPathContext() }, file.path)
    }

    storage.save("examples/out/allJavaFiles")
}