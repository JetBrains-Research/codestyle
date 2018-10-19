package codestyle.miner

import org.junit.Assert
import org.junit.Test

class IdExtractionTest {

    @Test
    fun testIdExtraction1() {
        val tree = parse("testData/idExtraction/1.java")

        val methodInfos = getMethodInfos(tree)

        Assert.assertEquals(1, methodInfos.size)
        Assert.assertEquals(MethodId("SingleFunction", "fun", setOf("int", "String[]")), methodInfos.first().id)
    }

    @Test
    fun testIdExtraction2() {
        val tree = parse("testData/idExtraction/2.java")

        val methodInfos = getMethodInfos(tree)

        Assert.assertEquals(1, methodInfos.size)
        Assert.assertEquals(MethodId("InnerClass", "main", setOf("String[]")), methodInfos.first().id)
    }

    @Test
    fun testIdExtraction3() {
        val tree = parse("testData/idExtraction/3.java")

        val methodInfos = getMethodInfos(tree)

        Assert.assertEquals(2, methodInfos.size)
        Assert.assertTrue(methodInfos.map { it.id }.contains(MethodId("InnerClass", "main", setOf("String[]"))))
        Assert.assertTrue(methodInfos.map { it.id }.contains(MethodId("SingleMethodInnerClass", "fun", setOf("String[]", "int"))))
    }
}