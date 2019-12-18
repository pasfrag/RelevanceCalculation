import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}

object main {

    def main(args: Array[String]): Unit = {
        val ss: SparkSession = SparkSession.builder().config("spark.master", "local[*]").getOrCreate()
        val sc: SparkContext = ss.sparkContext
        val sqlContext: SQLContext = ss.sqlContext

        val trainDF: DataFrame = ss.read.format("csv").option("header", "true").load("data/train.csv")
        val descDF: DataFrame = ss.read.format("csv").option("header", "true").load("data/product_descriptions.csv")
        val attrDF: DataFrame = ss.read.format("csv").option("header", "true").load("data/attributes.csv")

        /*trainDF.take(20).foreach(x => print(x))
        descDF.take(20).foreach(x => print(x))
        attrDF.take(20).foreach(x => print(x))*/

        trainDF.createOrReplaceTempView("train_desc")
        descDF.createOrReplaceTempView("desc_desc")
        attrDF.createOrReplaceTempView("attr_desc")

        val trainWithDescAndAttrDF:DataFrame = trainDF.join(descDF, descDF("product_uid") === trainDF("product_uid"))
        //.join(attrDF, attrDF("product_uid") === trainDF("product_uid"))

        //sqlContext.sql(
        //    """SELECT a.id, a.product_uid, a.product_title, a.search_term, a.relevance, b.product_description, c.name, c.value FROM train_desc as a INNER JOIN desc_desc as b on a.product_uid = b.product_uid INNER JOIN attr_desc as c on a.product_uid = c.product_uid GROUP BY""")

        trainWithDescAndAttrDF.take(50).foreach(x => print(x))
    }

}
