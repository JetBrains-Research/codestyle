package miningtool.impl.antlr.python

import me.vovak.antlr.parser.Python3Lexer
import me.vovak.antlr.parser.Python3Parser
import miningtool.common.Node
import miningtool.common.Parser
import org.antlr.v4.runtime.CommonTokenStream
import org.antlr.v4.runtime.RuleContext
import org.antlr.v4.runtime.UnbufferedCharStream
import java.io.InputStream


class PythonParser : Parser() {
    override fun parse(content: InputStream): Node? {
        val lexer = Python3Lexer(UnbufferedCharStream(content))
        val tokens = CommonTokenStream(lexer)
        val parser: Python3Parser = Python3Parser(tokens)

        val context = parser.file_input()

        return convert(context)
    }

    fun convert(context: RuleContext): Node? {
        println(context.parent)
        printContext(context)
        return null
    }

    fun printContext(context: RuleContext) {
        doPrintContext(context, 0)
    }

    fun doPrintContext(context: RuleContext, indent: Int) {
        val isTrivial = context.childCount == 1
        if (!isTrivial) {
            println(" ".repeat(indent) + Python3Parser.ruleNames[context.ruleIndex])
        }
        val newIndent = if (isTrivial) indent else indent + 1
        for (i in 0 until context.childCount) {
            doPrintContext(context.getChild(i) as RuleContext, newIndent)
        }
    }

}