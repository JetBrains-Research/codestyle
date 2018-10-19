package codestyle.miner

import com.github.gumtreediff.tree.ITree

data class BlobId(val id: String)

data class ChangeEntry(
        val commitId: String,
        val authorName: String,
        val authorEmail: String,
        val committerName: String,
        val committerEmail: String,
        val authorTime: Long,
        val committerTime: Long,
        val changeType: Char,
        val oldContentId: BlobId?,
        val newContentId: BlobId?,
        val oldPath: String?,
        val newPath: String?
)


data class MethodId(val enclosingClassName: String, val methodName: String, val argTypes: Set<String>)

data class MethodInfo(val node: ITree, val id: MethodId)

data class MethodMapping(val before: MethodInfo?, val after: MethodInfo?)