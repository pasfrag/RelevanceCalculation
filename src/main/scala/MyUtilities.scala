import org.apache.spark.ml.linalg.{SparseVector, Vector}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.sources.And

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
        else a.intersect(b).length.toDouble / a.union(b).length.toDouble
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

    def convertForCluster(rel: Double): Int = {
        if (rel < 2.0) 0
        else 1
    }

    def addSimilarities1(t: Double, d: Double, a: Double): Double = {
        0.5*t+0.3*d+0.2*a
    }

    def addSimilarities2(t: Double, d: Double, a: Double): Double = {
        0.6*t+0.3*d+0.1*a
    }

    def addSimilarities3(t: Double, d: Double, a: Double): Double = {
        t+d+a
    }

    def testArray(i: Int): Array[String] = {
        if (i==0) Array("title_tfidf", "search_tfidf")
        else if (i==1) Array("title_w2v", "search_w2v")
        else if (i==2) Array("cos_title_t")
        else if (i==3) Array("cos_title_w")
        else if (i==4) Array("jacc_title")
        else if (i==5) Array("cos_title_t", "title_w2v", "search_w2v")
        else if (i==6) Array("cos_title_t", "title_w2v", "search_w2v", "jacc_title")
        else if (i==7) Array("cos_title_t", "title_w2v", "search_w2v", "cos_title_w")
        else if (i==8) Array("cos_title_w", "title_tfidf", "search_tfidf")
        else if (i==9) Array("jacc_title", "title_tfidf", "search_tfidf", "title_w2v", "search_w2v")
        else if (i==10) Array("cos_title_t", "cos_title_w", "jacc_title")
        else if (i==11) Array("cos_title_t", "cos_desc_t", "cos_title_w", "cos_desc_w", "jacc_title", "jacc_desc")
        else if (i==12) Array("cos_title_t", "cos_desc_t", "cos_attr_t", "cos_title_w", "cos_desc_w", "cos_attr_w",
            "jacc_title", "jacc_desc", "jacc_attr")
        else if (i==13) Array("cos_title_t", "cos_desc_t", "cos_attr_t", "cos_title_w", "cos_desc_w", "cos_attr_w",
            "jacc_title", "jacc_desc", "jacc_attr", "comm_title")
        else if (i==14) Array("cos_title_t", "cos_desc_t", "cos_attr_t", "cos_title_w", "cos_desc_w", "cos_attr_w",
            "jacc_title", "jacc_desc", "jacc_attr", "comm_title", "comm_desc")
        else if (i==15) Array("cos_title_t", "cos_desc_t", "cos_attr_t", "cos_title_w", "cos_desc_w", "cos_attr_w",
            "jacc_title", "jacc_desc", "jacc_attr", "comm_title", "comm_desc", "comm_attr")
        else if (i==16) Array("cos_title_t", "cos_desc_t", "cos_attr_t", "cos_title_w", "cos_desc_w", "cos_attr_w",
            "jacc_title", "jacc_desc", "jacc_attr", "query_length")
        else if (i==17) Array("cos_title_t", "cos_desc_t", "cos_attr_t", "cos_title_w", "cos_desc_w", "cos_attr_w",
            "jacc_title", "jacc_desc", "jacc_attr", "title_ratio")
        else if (i==18) Array("cos_title_t", "cos_desc_t", "cos_attr_t", "cos_title_w", "cos_desc_w", "cos_attr_w",
            "jacc_title", "jacc_desc", "jacc_attr", "title_ratio", "desc_ratio")
        else if (i==19) Array("cos_title_t", "cos_desc_t", "cos_attr_t", "cos_title_w", "cos_desc_w", "cos_attr_w",
            "jacc_title", "jacc_desc", "jacc_attr", "title_ratio", "desc_ratio", "attr_ratio")
        else if (i==20) Array("cos_title_t", "cos_desc_t", "cos_attr_t", "cos_title_w", "cos_desc_w", "cos_attr_w",
            "jacc_title", "jacc_desc", "jacc_attr", "comm_title", "comm_desc", "comm_attr", "query_length",
            "title_ratio", "desc_ratio", "attr_ratio")
        else if (i==21) Array("cos_title_t", "cos_title_w", "jacc_title", "title_w2v", "search_w2v")
        else if (i==22) Array("cos_title_t", "cos_desc_t", "cos_attr_t", "cos_title_w", "cos_desc_w", "cos_attr_w",
            "jacc_title", "jacc_desc", "jacc_attr", "comm_title", "comm_desc", "comm_attr", "query_length",
            "title_ratio", "desc_ratio", "attr_ratio", "title_w2v", "search_w2v")
        else if (i==23) Array("cos_title_t", "title_tfidf", "search_tfidf")
        else if (i==24) Array("cos_title_w", "title_w2v", "search_w2v")
        else if (i==25) Array("cos_title_t", "cos_title_w", "jacc_title", "title_tfidf", "search_tfidf")
        else Array()
    }

    def customPrediction(d: Double): Int = {
        if( d > 0.1) 1 else 0
    }

}
