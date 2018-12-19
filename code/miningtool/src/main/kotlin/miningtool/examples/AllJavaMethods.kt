package miningtool.examples

import miningtool.common.toPathContext
import miningtool.parse.java.GumTreeJavaParser
import miningtool.parse.java.GumTreeMethodSplitter
import miningtool.parse.java.MethodInfo
import miningtool.parse.java.getMethodInfo
import miningtool.paths.PathMiner
import miningtool.paths.PathRetrievalSettings
import miningtool.paths.storage.VocabularyPathStorage
import java.io.File

// Retrieve paths for all methods in all Java files in the folder
private fun getCsvFriendlyMethodId(methodInfo: MethodInfo?): String {
    if (methodInfo == null) return "unknown_method"
    return "${methodInfo.enclosingClassName}.${methodInfo.methodName}(${methodInfo.parameterTypes.joinToString("|")})"
}

fun allJavaMethods() {
    val folder = "./testData"

    val miner = PathMiner(PathRetrievalSettings(5, 5))

    val storage = VocabularyPathStorage()

    File(folder).walkTopDown().filter { it.path.endsWith(".java") }.forEach { file ->
        //parse file
        val fileNode = GumTreeJavaParser().parse(file.inputStream()) ?: return@forEach

        //extract method nodes
        val methodNodes = GumTreeMethodSplitter().split(fileNode)

        methodNodes.forEach {
            //Retrieve paths from every node individually
            val paths = miner.retrievePaths(it)
            //Retrieve a method identifier
            val entityId = "${file.path}::${getCsvFriendlyMethodId(it.getMethodInfo())}"
            storage.store(paths.map { it.toPathContext() }, entityId)
        }
    }

    storage.save("examples/out/allJavaMethods")
}