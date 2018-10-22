package codestyle.miner

import com.github.gumtreediff.client.Run
import com.github.gumtreediff.matchers.Matchers
import com.github.gumtreediff.tree.TreeContext

fun getMethodMappings(treeBefore: TreeContext, treeAfter: TreeContext): Collection<MethodMapping> {
    val infosBefore = getMethodInfos(treeBefore)
    val infosAfter = getMethodInfos(treeAfter)

    Run.initGenerators()

    val matcher = Matchers.getInstance().getMatcher(treeBefore.root, treeAfter.root)
    matcher.match()
    val gtMappings = matcher.mappings

    val mappings: MutableSet<MethodMapping> = HashSet()

    infosBefore.forEach {
        val dst = gtMappings.getDst(it.node)
        if (dst == null) {
            mappings.add(MethodMapping(it, null))
            return@forEach
        }
        val dstInfo = infosAfter.firstOrNull { info -> info.node == dst }
        if (dstInfo == null) {
            println("Method node $it mapped to unknown node $dst by GumTree")
            return@forEach
        }

        mappings.add(MethodMapping(it, dstInfo))
    }

    infosAfter.forEach {
        val src = gtMappings.getSrc(it.node)
        if (src == null) {
            mappings.add(MethodMapping(null, it))
        }
    }

    return mappings
}

fun getMethodMappings(filenameBefore: String, filenameAfter: String): Collection<MethodMapping> {
    return getMethodMappings(parse(filenameBefore), parse(filenameAfter))
}