package miningtool.paths

import miningtool.common.Node
import miningtool.common.postOrder
import org.junit.Assert
import org.junit.Test

abstract class PathWorkerTestBase {
    abstract fun getTree(): Node

    @Test
    fun anyPathsReturned() {
        val allPaths = PathWorker().retrievePaths(getTree(), PathRetrievalSettings.NO_LIMIT)
        Assert.assertFalse("At least some paths should be retrieved from a non-trivial tree", allPaths.isEmpty())
    }

    @Test
    fun pathsCountNoLimit() {
        val tree = getTree()
        val nLeaves = tree.postOrder().count { it.isLeaf() }

        val allPaths = PathWorker().retrievePaths(tree, PathRetrievalSettings.NO_LIMIT)
        val expectedCount = (nLeaves * (nLeaves - 1)) / 2

        Assert.assertEquals("A tree with $nLeaves leaves contains $expectedCount paths, " +
                "one per distinct ordered pair of leaves. Worker returned ${allPaths.size}",
                expectedCount, allPaths.size)
    }

    @Test
    fun pathsCountWidth1() {
        val tree = getTree()
        val nLeaves = tree.postOrder().count { it.isLeaf() }

        val allPaths = PathWorker().retrievePaths(tree, PathRetrievalSettings(Int.MAX_VALUE, 1))
        val expectedCount = nLeaves - 1

        Assert.assertEquals("A tree with $nLeaves leaves contains $expectedCount paths of width 1. " +
                "Worker returned ${allPaths.size}",
                expectedCount, allPaths.size)
    }

    @Test
    fun pathsCountAnyWidth() {
        val tree = getTree()
        val nLeaves = tree.postOrder().count { it.isLeaf() }

        for (maxWidth in 1..nLeaves) {
            val paths = PathWorker().retrievePaths(tree, PathRetrievalSettings(Int.MAX_VALUE, maxWidth))
            val expectedPathsCount = getPathsCountWithNoHeightLimit(nLeaves, maxWidth)
            Assert.assertEquals("A tree with $nLeaves nodes should contain $expectedPathsCount paths " +
                    "of width up to $maxWidth, worker returned ${paths.size}",
                    expectedPathsCount, paths.size)
        }
    }

    @Test
    fun pathValidity() {
        val tree = getTree()

        val allPaths = PathWorker().retrievePaths(tree, PathRetrievalSettings.NO_LIMIT)
        allPaths.forEach {
            assertPathIsValid(it)
        }
    }

    @Test
    fun countFunctionsMatch() {
        val tree = getTree()
        val leavesCount = tree.postOrder().count { it.isLeaf() }

        for (maxWidth in 1..leavesCount) {
            Assert.assertEquals(getPathsCountWithNoHeightLimit(leavesCount, maxWidth),
                    countPossiblePaths(tree, Int.MAX_VALUE, maxWidth))
        }
    }

    @Test
    fun countsForAllLimitCombinations() {
        val maxHeightLimit = 100

        val tree = getTree()
        val leavesCount = tree.postOrder().count { it.isLeaf() }

        for (maxHeight in 1..maxHeightLimit) {
            for (maxWidth in 1..leavesCount) {
                Assert.assertEquals(
                        countPossiblePaths(tree, maxHeight, maxWidth),
                        PathWorker().retrievePaths(tree, PathRetrievalSettings(maxHeight, maxWidth)))
            }
        }
    }

    @Test
    fun validityForAllLimitCombinations() {
        val maxHeightLimit = 100

        val tree = getTree()
        val leavesCount = tree.postOrder().count { it.isLeaf() }

        for (maxHeight in 1..maxHeightLimit) {
            for (maxWidth in 1..leavesCount) {
                val paths = PathWorker().retrievePaths(tree, PathRetrievalSettings(maxHeight, maxWidth))
                paths.forEach { assertPathIsValid(it) }
            }
        }
    }
}