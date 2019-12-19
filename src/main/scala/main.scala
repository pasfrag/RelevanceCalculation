import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.sql.{Column, DataFrame, SQLContext, SparkSession}

object main {

    def main(args: Array[String]): Unit = {

        // Starting the session
        val spark: SparkSession = SparkSession.builder().config("spark.master", "local[*]").getOrCreate()
        val sc: SparkContext = spark.sparkContext
        val sqlContext: SQLContext = spark.sqlContext

        // Reading the data
        val trainDF: DataFrame = spark.read.format("csv").option("header", "true").load("data/train.csv")
        val descDF: DataFrame = spark.read.format("csv").option("header", "true").load("data/product_descriptions.csv")
        val attrDF: DataFrame = spark.read.format("csv").option("header", "true").load("data/attributes.csv")

        // Tokenizing the training data
        val tokenizer = new Tokenizer().setInputCol("product_title").setOutputCol("title_tokenized")
        val semiTokenizedTrainDF = tokenizer.transform(trainDF)
        tokenizer.setInputCol("search_term").setOutputCol("search_tokenized")
        val tokenizedTrainDF = tokenizer.transform(semiTokenizedTrainDF)
        val columnsToRemove = Seq("id", "product_title", "search_term")
        val removedTrainDF = tokenizedTrainDF.select(tokenizedTrainDF.columns
          .filter(colName => !columnsToRemove.contains(colName))
          .map(colName => new Column(colName)): _*
        )

        // Tokenizing the description data
        tokenizer.setInputCol("product_description").setOutputCol("description_tokenized")
        val tokenizedDescriptionDF = tokenizer.transform(descDF)
        val removedDescriptionDF = tokenizedDescriptionDF.select(tokenizedDescriptionDF.columns
          .filter(colName => colName != "product_description")
          .map(colName => new Column(colName)): _*
        )

        // Tokenizing the attribute data
        tokenizer.setInputCol("name").setOutputCol("name_tokenized")
        val semiTokenizedAttributeDF = tokenizer.transform(attrDF)
        tokenizer.setInputCol("value").setOutputCol("value_tokenized")
        val tokenizedAttributeDF = tokenizer.transform(semiTokenizedAttributeDF)
        val columnsToRemove1 = Seq("name", "value")
        val removedAttributeDF = tokenizedAttributeDF.select(tokenizedAttributeDF.columns
          .filter(colName => !columnsToRemove1.contains(colName))
          .map(colName => new Column(colName)): _*
        )
        removedAttributeDF.printSchema()

//        trainDF.take(20).foreach(x => print(x))
//        descDF.take(20).foreach(x => print(x))
//        attrDF.take(20).foreach(x => print(x))
//
//        trainDF.createOrReplaceTempView("train_desc")
//        descDF.createOrReplaceTempView("desc_desc")
//        attrDF.createOrReplaceTempView("attr_desc")
//
//        val trainWithDescAndAttrDF:DataFrame = trainDF.join(descDF, descDF("product_uid") === trainDF("product_uid"))
//            .join(attrDF, attrDF("product_uid") === trainDF("product_uid"))
//
//        sqlContext.sql(
//            """SELECT a.id, a.product_uid, a.product_title, a.search_term, a.relevance, b.product_description, c.name, c.value FROM train_desc as a INNER JOIN desc_desc as b on a.product_uid = b.product_uid INNER JOIN attr_desc as c on a.product_uid = c.product_uid GROUP BY""")
//
//        trainWithDescAndAttrDF.take(50).foreach(x => print(x))
    }

}
