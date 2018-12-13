package miningtool.examples

import miningtool.common.Parser
import miningtool.parse.antlr.java.JavaParser
import miningtool.parse.java.GumTreeJavaParser
import miningtool.paths.PathMiner
import miningtool.paths.PathRetrievalSettings
import java.io.File


fun allJavaFiles() {
    val folder = "./testData"

    val parserProvider: () -> Parser = { JavaParser() }
    val miner = PathMiner(parserProvider, PathRetrievalSettings(5, 5))

    File(folder).walkTopDown().filter { it.path.endsWith(".java") }.forEach {
        val paths = miner.retrievePaths(it.inputStream())
        println(it.path)
        println(paths.size)

        paths.forEach { path ->
            println(path.upwardNodes.first().getToken() +
                    path.upwardNodes.map { it.getTypeLabel() }.toString() +
                    path.downwardNodes.map { it.getTypeLabel() }.toString() +
                    path.downwardNodes.last().getToken())}
    }
}