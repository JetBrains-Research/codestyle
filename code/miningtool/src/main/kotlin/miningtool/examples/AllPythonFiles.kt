package miningtool.examples

import miningtool.parse.antlr.python.PythonParser
import miningtool.paths.PathMiner
import miningtool.paths.PathRetrievalSettings
import java.io.File


fun allPythonFiles() {
    val folder = "."

    val miner = PathMiner(PathRetrievalSettings(5, 5))

    val parser = PythonParser()

    File(folder).walkTopDown().filter { it.path.endsWith(".py") }.forEach {
        val node = parser.parse(it.inputStream()) ?: return@forEach
        val paths = miner.retrievePaths(node)

        paths.forEach { path ->
            println(path.upwardNodes.first().getToken() +
                    path.upwardNodes.map { it.getTypeLabel() }.toString() +
                    path.downwardNodes.map { it.getTypeLabel() }.toString() +
                    path.downwardNodes.last().getToken())
        }
    }
}