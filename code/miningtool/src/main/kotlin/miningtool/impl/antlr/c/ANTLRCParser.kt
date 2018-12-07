package miningtool.impl.antlr.c

import me.vovak.antlr.parser.CLexer
import me.vovak.antlr.parser.CParser
import miningtool.common.Node
import miningtool.common.Parser
import miningtool.impl.antlr.convertAntlrTree
import org.antlr.v4.runtime.ANTLRInputStream
import org.antlr.v4.runtime.CommonTokenStream
import java.io.InputStream

class ANTLRCParser: Parser() {
    override fun parse(content: InputStream): Node? {
        val lexer = CLexer(ANTLRInputStream(content))
        val tokens = CommonTokenStream(lexer)
        val parser = CParser(tokens)
        val context = parser.compilationUnit()
        return convertAntlrTree(context, CParser.ruleNames)
    }
}