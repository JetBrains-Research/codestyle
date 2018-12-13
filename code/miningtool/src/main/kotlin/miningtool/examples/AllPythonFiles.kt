
package miningtool.examples

import miningtool.common.Parser
import miningtool.parse.antlr.python.PythonParser
import miningtool.paths.PathMiner
import miningtool.paths.PathRetrievalSettings
import java.io.File


fun allPythonFiles() {
    val folder = "."

    val parserProvider: () -> Parser = { PythonParser() }
    val miner = PathMiner(parserProvider, PathRetrievalSettings(5, 5))

    File(folder).walkTopDown().filter { it.path.endsWith(".py") }.forEach {
        val paths = miner.retrievePaths(it.inputStream())
        println(it.path)
        println(paths.size)

        paths.forEach { path ->
            println(path.upwardNodes.first().getToken() +
                    path.upwardNodes.map { it.getTypeLabel() }.toString() +
                    path.downwardNodes.map { it.getTypeLabel() }.toString() +
                    path.downwardNodes.last().getToken())
        }
    }
}