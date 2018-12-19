package miningtool

import miningtool.examples.allJavaFiles
import miningtool.examples.allJavaMethods
import miningtool.examples.allPythonFiles

fun main(args: Array<String>) {
    runExamples()
}

fun runExamples() {
    allJavaFiles()
    allJavaMethods()
    allPythonFiles()
}