package codestyle.miner

import com.github.gumtreediff.tree.ITree
import java.io.File
import java.io.FileInputStream


fun getMethodContent(node: ITree?, blobId: BlobId?, repoName: String): String? {
    if (node == null || blobId == null) {
        return null
    }

    val startOffset = node.pos
    val endOffset = node.endPos

    val filename = getBlobPath(blobId.id, repoName)

    return readContent(filename, startOffset, endOffset)
}

fun readContent(filename: String, startOffset: Int, endOffset: Int): String {
    val input = FileInputStream(File(filename))
    input.skip(startOffset.toLong())

    val len = endOffset - startOffset
    val buffer = ByteArray(len)

    input.read(buffer)

//    println("+++++")

    input.close()

    return String(buffer)
}

fun initMethodContentsDir(repoName: String) {
    val dirPath = "out/$repoName/methodContents"
    File(dirPath).mkdirs()
}

fun saveMethodContent(id: String?, contents: String?, repoName: String) {
    if (id == null || contents == null) return
    val path = "out/$repoName/methodContents/$id"
    File(path).printWriter().use {
        it.write(contents)
    }
}