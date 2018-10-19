package codestyle.miner

import com.github.gumtreediff.gen.jdt.JdtTreeGenerator
import com.github.gumtreediff.tree.ITree
import com.github.gumtreediff.tree.TreeContext

fun readAndParseBlob(id: String): TreeContext {
    val file = "../python-miner/data/exploded/intellij-community/blobs/$id"
    return parse(file)
}

fun parse(file: String): TreeContext {
    return JdtTreeGenerator().generateFromFile(file)
}

fun getEnclosingClassName(methodNode: ITree, context: TreeContext): String {
    val classDeclarationNode = methodNode.parents.firstOrNull { context.getTypeLabel(it.type) == "TypeDeclaration" } ?: return ""
    val nameNode = classDeclarationNode.children.firstOrNull { context.getTypeLabel(it.type) == "SimpleName" }
    return nameNode?.label ?: ""
}

fun getMethodName(methodNode: ITree, context: TreeContext): String {
    val nameNode = methodNode.children.firstOrNull { context.getTypeLabel(it.type) == "SimpleName" }
    return nameNode?.label ?: ""
}

fun getParameterTypes(methodNode: ITree, context: TreeContext): Set<String> {
    val result: MutableSet<String> = HashSet()
    val argDeclarationNodes = methodNode.children
            .filter { context.getTypeLabel(it.type) == "SingleVariableDeclaration" }
    argDeclarationNodes.forEach {
        val typeNode = it.children.filter { c -> context.getTypeLabel(c.type).endsWith("Type") }.firstOrNull()
        if (typeNode != null) result.add(typeNode.label)
    }
    return result
}

fun getMethodId(methodNode: ITree, context: TreeContext): MethodId {
    return MethodId(
            getEnclosingClassName(methodNode, context),
            getMethodName(methodNode, context),
            getParameterTypes(methodNode, context)
    )
}

fun getMethodInfo(methodNode: ITree, context: TreeContext): MethodInfo {
    return MethodInfo(methodNode, getMethodId(methodNode, context))
}

fun getMethodInfos(treeContext: TreeContext): Collection<MethodInfo> {
    val methodNodes = treeContext.root.descendants
            .filter { treeContext.getTypeLabel(it.type) == "MethodDeclaration" }
    return methodNodes.map { getMethodInfo(it, treeContext) }
}
