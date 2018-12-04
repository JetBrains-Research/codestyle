package miningtool.impl.antlr.python

import me.vovak.antlr.parser.Python3Lexer
import me.vovak.antlr.parser.Python3Parser
import miningtool.common.Node
import miningtool.common.Parser
import miningtool.impl.antlr.convertAntlrTree
import org.antlr.v4.runtime.ANTLRInputStream
import org.antlr.v4.runtime.CommonTokenStream
import org.antlr.v4.runtime.ParserRuleContext
import org.antlr.v4.runtime.RuleContext
import org.antlr.v4.runtime.tree.ErrorNode
import org.antlr.v4.runtime.tree.ParseTreeListener
import org.antlr.v4.runtime.tree.ParseTreeWalker
import org.antlr.v4.runtime.tree.TerminalNode
import java.io.InputStream

class PrintListener : ParseTreeListener {
    override fun enterEveryRule(ctx: ParserRuleContext?) {
        if (ctx == null) return
//        if (ctx.children.size == 1) return
        println(Python3Parser.ruleNames[ctx.ruleIndex] + " (${ctx.children.size} children)")
    }

    override fun exitEveryRule(ctx: ParserRuleContext?) {
    }

    override fun visitErrorNode(node: ErrorNode?) {
        if (node == null) return
        println("ERR " + node.symbol.text)
    }

    override fun visitTerminal(node: TerminalNode?) {
        if (node == null) return
        println("TERM " + node.symbol.text)
    }

}

class PythonParser : Parser() {
    override fun parse(content: InputStream): Node? {
        val lexer = Python3Lexer(ANTLRInputStream(content))
        val tokens = CommonTokenStream(lexer)
        val parser: Python3Parser = Python3Parser(tokens)

        val context = parser.file_input()

        ParseTreeWalker().walk(PrintListener(), context)

        return convert(context)
    }

    fun convert(context: RuleContext): Node? {
        println(context.parent)
        printContext(context)
        //TODO

        return convertAntlrTree(context as ParserRuleContext, Python3Parser.ruleNames)
    }

    fun printContext(context: RuleContext) {
        doPrintContext(context, 0)
    }

    fun doPrintContext(context: RuleContext, indent: Int) {
        val isTrivial = context.childCount == 1
        if (!isTrivial) {
            println(" ".repeat(indent) + Python3Parser.ruleNames[context.ruleIndex] + " " + context.text)
        }
        val newIndent = if (isTrivial) indent else indent + 1
        for (i in 0 until context.childCount) {
            val child = context.getChild(i) as? RuleContext ?: continue
            doPrintContext(child, newIndent)
        }
    }

}