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


    //New method addition
    @Test
    fun testMatching3() {
        val fileBefore = "testData/differ/matching/3/Before.java"
        val fileAfter = "testData/differ/matching/3/After.java"

        val mappings = getMethodMappings(fileBefore, fileAfter)

        Assert.assertEquals(2, mappings.size)

        val idPairs = mappings.map { Pair(it.before?.id, it.after?.id) }

        Assert.assertTrue(idPairs.contains(Pair(
                MethodId("SingleFunction", "fun1", setOf("String[]", "int")),
                MethodId("TwoFunctions", "fun1", setOf("String[]", "int"))
        )))

        Assert.assertTrue(idPairs.contains(Pair(
                null,
                MethodId("TwoFunctions", "fun2", setOf("int"))
        )))

    }

    //Existing method removal
    @Test
    fun testMatching4() {
        val fileBefore = "testData/differ/matching/4/Before.java"
        val fileAfter = "testData/differ/matching/4/After.java"

        val mappings = getMethodMappings(fileBefore, fileAfter)

        Assert.assertEquals(2, mappings.size)

        val idPairs = mappings.map { Pair(it.before?.id, it.after?.id) }

        Assert.assertTrue(idPairs.contains(Pair(
                MethodId("TwoFunctions", "fun1", setOf("String[]", "int")),
                MethodId("SingleFunction", "fun1", setOf("String[]", "int"))
        )))

        Assert.assertTrue(idPairs.contains(Pair(
                MethodId("TwoFunctions", "fun2", setOf("int")),
                null
        )))
    }
}