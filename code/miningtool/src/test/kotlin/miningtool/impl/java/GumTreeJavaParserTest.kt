package miningtool.impl.java

import org.junit.Assert
import org.junit.Test
import java.io.File
import java.io.FileReader

class GumTreeJavaParserTest {
    @Test
    fun testNodeIsNotNull() {
        val parser = GumTreeJavaParser()
        val file = File("testData/1.java")

        val node = parser.parse(FileReader(file))
        Assert.assertNotNull("Parse tree for a valid file should not be null", node)
    }
}