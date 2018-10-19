package codestyle.miner

import org.junit.Assert
import org.junit.Test

class MethodMatchingTest {
    @Test
    fun testMatching1() {
        val fileBefore = "testData/differ/matching/1/Before.java"
        val fileAfter = "testData/differ/matching/1/After.java"

        val mappings = getMethodMappings(fileBefore, fileAfter)

        Assert.assertEquals(1, mappings.size)
        Assert.assertNotNull(mappings.first().before)
        Assert.assertNotNull(mappings.first().after)
        Assert.assertEquals(mappings.first().before!!.id, MethodId("SingleFunction", "main", setOf("String[]")))
        Assert.assertEquals(mappings.first().after!!.id, MethodId("SingleFunction", "main", setOf("String[]")))
    }

    @Test
    fun testMatching2() {
        val fileBefore = "testData/differ/matching/2/Before.java"
        val fileAfter = "testData/differ/matching/2/After.java"

        val mappings = getMethodMappings(fileBefore, fileAfter)

        Assert.assertEquals(1, mappings.size)
        Assert.assertNotNull(mappings.first().before)
        Assert.assertNotNull(mappings.first().after)
        Assert.assertEquals(mappings.first().before!!.id, MethodId("SingleFunction", "fun", setOf("String[]", "int")))
        Assert.assertEquals(mappings.first().after!!.id, MethodId("SingleFunction", "main", setOf("String[]")))
    }
}