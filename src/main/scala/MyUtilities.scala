import breeze.numerics.sqrt
import org.apache.spark.ml.linalg.{SparseVector, Vector}

import scala.collection.mutable

object MyUtilities {

    def cosineSimilarity (vectorA: SparseVector, vectorB:SparseVector/*,normASqrt:Double,normBSqrt:Double*/): Double = {
        if (vectorA==null | vectorB==null) 0
        else{
            val normASqrt = norm(vectorA)
            val normBSqrt = norm(vectorB)
            var dotProduct = 0.0
            for (i <-  vectorA.indices){ dotProduct += vectorA(i) * vectorB(i) }
            val div = normASqrt * normBSqrt
            if (div == 0)  0
            else dotProduct / div}
    }

    def jaccardSimilarity (a: mutable.WrappedArray[String], b: mutable.WrappedArray[String]): Double = {
        if (a == null | b == null) 0
        else (a.intersect(b).length).toDouble / (a.union(b).length).toDouble
    }

    def toSparse (a : Vector): Vector = {
        if (a == null) null
        else a.toSparse
    }

    def toDouble1 (s: String): Double = {
        s.toDouble
    }

    def norm (vector: SparseVector): Double = {
        if (vector == null) 0
        else{
            var dotProduct = 0.0
            for (i <-  vector.indices){ dotProduct += vector(i) * vector(i) }
            val returnVar = math.sqrt(dotProduct)
            returnVar
        }
    }

    // Replace null
    def replaceNull (a: mutable.WrappedArray[String]): mutable.WrappedArray[String] = {
        if(a == null) Array("")
        else a
    }

    // Ratio
    def ratio (commonWords: Int, length: Int): Double = {
        if(length==0) 0
        else{commonWords.toDouble/length.toDouble}
    }

    // Common words
    def commonWords (a: mutable.WrappedArray[String], b: mutable.WrappedArray[String]): Int = {
        if (a==null | b==null) 0
        else{ a.intersect(b).length }
    }

    // Lenth of querry
    def queryLength(a: mutable.WrappedArray[String]): Int = {
        if (a==null) 0
        else {a.size}
    }



}
