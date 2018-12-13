package miningtool.parse.antlr.python

import me.vovak.antlr.parser.Python3Lexer
import me.vovak.antlr.parser.Python3Parser
import miningtool.common.Node
import miningtool.common.Parser
import miningtool.parse.antlr.convertAntlrTree
import org.antlr.v4.runtime.ANTLRInputStream
import org.antlr.v4.runtime.CommonTokenStream
import java.io.InputStream

class PythonParser : Parser() {
    override fun parse(content: InputStream): Node? {
        val lexer = Python3Lexer(ANTLRInputStream(content))
        val tokens = CommonTokenStream(lexer)
        val parser = Python3Parser(tokens)

        val context = parser.file_input()

        return convertAntlrTree(context, Python3Parser.ruleNames)
    }

}