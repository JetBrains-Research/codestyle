package miningtool.paths

import miningtool.common.ASTPath
import miningtool.common.Node
import miningtool.common.Parser
import java.io.InputStream


data class PathRetrievalSettings(val maxHeight: Int, val maxWidth: Int) {
}

class PathMiner(val parserProvider: () -> Parser, val settings: PathRetrievalSettings) {
    private val pathWorker = PathWorker()
    fun retrievePaths(input: InputStream): Collection<ASTPath> {
        val parser = parserProvider.invoke()
        val tree = parser.parse(input) ?: return emptyList() //todo verbose exceptions, option to handle parse errors

        return doRetrievePaths(tree)
    }

    private fun doRetrievePaths(tree: Node): Collection<ASTPath> {
        //todo adapt from the custom miner code
        return pathWorker.retrievePaths(tree, settings.maxHeight, settings.maxWidth)
    }
}