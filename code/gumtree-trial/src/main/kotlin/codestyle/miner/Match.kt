package codestyle.miner

import com.github.gumtreediff.tree.TreeContext

fun getMethodMappings(treeBefore: TreeContext, treeAfter: TreeContext): Collection<MethodMapping> {
    throw NotImplementedError("Not implemented yet")
}

fun getMethodMappings(filenameBefore: String, filenameAfter: String): Collection<MethodMapping> {
    return getMethodMappings(parse(filenameBefore), parse(filenameAfter))
}