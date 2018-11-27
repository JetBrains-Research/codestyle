package miningtool.impl.java

import com.github.gumtreediff.client.Run
import com.github.gumtreediff.gen.jdt.JdtTreeGenerator
import com.github.gumtreediff.tree.TreeContext
import miningtool.common.Node
import miningtool.common.Parser
import java.io.Reader

class GumTreeJavaParser : Parser() {
    init {
        Run.initGenerators()
    }

    override fun parse(content: Reader): Node? {
        val treeContext = JdtTreeGenerator().generate(content)
        return wrapGumTreeNode(treeContext)
    }
}

fun wrapGumTreeNode(treeContext: TreeContext): GumTreeJavaNode {
    return GumTreeJavaNode(treeContext.root, treeContext, null)
}