package miningtool.impl.antlr

import miningtool.common.Node
import org.antlr.v4.runtime.ParserRuleContext
import org.antlr.v4.runtime.tree.ErrorNode
import org.antlr.v4.runtime.tree.TerminalNode

class SimpleNode(private val myTypeLabel: String, private val myParent: Node?, private val myToken: String?): Node {
    private var myChildren: List<Node> = emptyList()

    fun setChildren(newChildren: List<Node>) {
        myChildren = newChildren
    }

    override fun getTypeLabel(): String {
        return myTypeLabel
    }

    override fun getChildren(): List<Node> {
        return myChildren
    }

    override fun getParent(): Node? {
        return myParent
    }

    override fun getToken(): String {
        return myToken?: "null"
    }

    override fun isLeaf(): Boolean {
        return myChildren.isEmpty()
    }
}

fun convertAntlrTree(tree: ParserRuleContext, ruleNames: Array<String>): SimpleNode {
    return simplifyTree(convertRuleContext(tree, ruleNames, null))
}

private fun convertRuleContext(ruleContext: ParserRuleContext, ruleNames: Array<String>, parent: Node?): SimpleNode {
    val typeLabel = ruleNames[ruleContext.ruleIndex]
    val currentNode = SimpleNode(typeLabel, parent, null)
    val children: MutableList<Node> = ArrayList()

    ruleContext.children.forEach {
        if (it is TerminalNode) {
            children.add(convertTerminal(it, currentNode))
            return@forEach
        }
        if (it is ErrorNode) {
            children.add(convertErrorNode(it, currentNode))
            return@forEach
        }
        children.add(convertRuleContext(it as ParserRuleContext, ruleNames, currentNode))
    }
    currentNode.setChildren(children)

    return currentNode
}

private fun convertTerminal(terminalNode: TerminalNode, parent: Node?): SimpleNode {
    return SimpleNode("Terminal", parent, terminalNode.symbol.text)
}

private fun convertErrorNode(errorNode: ErrorNode, parent: Node?): SimpleNode {
    return SimpleNode("Error", parent, errorNode.text)
}

private fun simplifyTree(tree: SimpleNode): SimpleNode {
    return if (tree.getChildren().size == 1) {
        simplifyTree(tree.getChildren().first() as SimpleNode)
    } else {
        tree.setChildren(tree.getChildren().map { simplifyTree(it as SimpleNode) })
        tree
    }
}

