package codestyle.miner

import org.junit.Assert
import org.junit.Test

class MethodMatchingTest {
    @Test
    fun testMatching1() {
        val fileBefore = "testData/differ/matching/1/Before.java"
        val fileAfter = "testData/differ/matching/1/After.java"

        val context = getMappingContext(fileBefore, fileAfter)

        Assert.assertEquals(1, context.mappings.size)
        Assert.assertNotNull(context.mappings.first().before)
        Assert.assertNotNull(context.mappings.first().after)
        Assert.assertEquals(context.mappings.first().before!!.id, MethodId("SingleFunction", "main", setOf("String[]")))
        Assert.assertEquals(context.mappings.first().after!!.id, MethodId("SingleFunction", "main", setOf("String[]")))
    }

    @Test
    fun testMatching2() {
        val fileBefore = "testData/differ/matching/2/Before.java"
        val fileAfter = "testData/differ/matching/2/After.java"

        val context = getMappingContext(fileBefore, fileAfter)

        Assert.assertEquals(1, context.mappings.size)
        Assert.assertNotNull(context.mappings.first().before)
        Assert.assertNotNull(context.mappings.first().after)
        Assert.assertEquals(context.mappings.first().before!!.id, MethodId("SingleFunction", "fun", setOf("String[]", "int")))
        Assert.assertEquals(context.mappings.first().after!!.id, MethodId("SingleFunction", "main", setOf("String[]")))
    }


    //New method addition
    @Test
    fun testMatching3() {
        val fileBefore = "testData/differ/matching/3/Before.java"
        val fileAfter = "testData/differ/matching/3/After.java"

        val context = getMappingContext(fileBefore, fileAfter)

        Assert.assertEquals(2, context.mappings.size)

        val idPairs = context.mappings.map { Pair(it.before?.id, it.after?.id) }

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

        val context = getMappingContext(fileBefore, fileAfter)

        Assert.assertEquals(2, context.mappings.size)

        val idPairs = context.mappings.map { Pair(it.before?.id, it.after?.id) }

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