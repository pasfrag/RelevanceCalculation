import org.apache.log4j.{Level, LogManager}
import org.apache.spark.SparkContext
import org.apache.spark.ml.clustering.{BisectingKMeans, GaussianMixture, KMeans, LDA}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, ClusteringEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.{HashingTF, IDF, StopWordsRemover, Tokenizer, VectorAssembler, Word2Vec}
import org.apache.spark.ml.linalg.{Matrices, Matrix, SparseVector, Vector}
import org.apache.spark.ml.regression.{DecisionTreeRegressor, GBTRegressor, IsotonicRegression, LinearRegression, RandomForestRegressor}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.feature.Stemmer
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}
import org.apache.spark.sql.functions.{col, collect_set, concat, concat_ws, lit, udf}
import org.apache.spark.sql.types.DoubleType

import scala.collection.mutable


object main {

    def main(args: Array[String]): Unit = {

        // Starting the session
        val spark: SparkSession = SparkSession.builder().config("spark.master", "local[*]").getOrCreate()
        import spark.implicits._
        val sc: SparkContext = spark.sparkContext
        val sqlContext: SQLContext = spark.sqlContext
        LogManager.getRootLogger.setLevel(Level.ERROR)

//        // Reading the data
//        val trainDF: DataFrame = spark.read.format("csv").option("header", "true").load("data/train.csv")
//        val descDF: DataFrame = spark.read.format("csv").option("header", "true").load("data/product_descriptions.csv")
//        val attrDF: DataFrame = spark.read.format("csv").option("header", "true").load("data/attributes.csv")
//
//        // General used variables
//        val tokenizer = new Tokenizer().setInputCol("product_title").setOutputCol("title_tokenized")
//        val remover = new StopWordsRemover()
//        val stemmer = new Stemmer().setLanguage("English")
//        val hashingTF = new HashingTF()
//        val idf = new IDF()
//        val word2Vec = new Word2Vec()//.setMaxIter(10)
//        val toSparse = udf[Vector, Vector](MyUtilities.toSparse)
//        val jaccardSimilarity = udf[Double, mutable.WrappedArray[String], mutable.WrappedArray[String]](MyUtilities.jaccardSimilarity)
//        val cosineSimilarity = udf[Double, SparseVector, SparseVector](MyUtilities.cosineSimilarity)
//        val ratio = udf[Double, Int, Int](MyUtilities.ratio)
//        val udfToDouble = udf[Double, String](MyUtilities.toDouble1)
//        val replaceNull = udf[mutable.WrappedArray[String], mutable.WrappedArray[String]](MyUtilities.replaceNull)
//        val commonWords = udf[Int, mutable.WrappedArray[String], mutable.WrappedArray[String]](MyUtilities.commonWords)
//        val queryLength = udf[Int, mutable.WrappedArray[String]](MyUtilities.queryLength)
        val toDual = udf[Int, Double](MyUtilities.convertForCluster)
//
//        //      -----------------------------Training Data Preprocess-----------------------------------------------
//        // Tokenizing the training data
//        val semiTokenizedTrainDF = tokenizer.transform(trainDF)
//        val tokenizedTrainDF = tokenizer.setInputCol("search_term").setOutputCol("search_tokenized")
//          .transform(semiTokenizedTrainDF)
//
//        //Removing stopwords from training data
//        val semiFilteredTrainDF = remover.setInputCol("title_tokenized").setOutputCol("title_filtered")
//          .transform(tokenizedTrainDF)
//        val filteredTrainDF = remover.setInputCol("search_tokenized").setOutputCol("search_filtered")
//          .transform(semiFilteredTrainDF)
//
//
//        // Stemming Training data
//        val semiStemmedTrainDF = stemmer.setInputCol("title_filtered").setOutputCol("title_stemmed")
//          .transform(filteredTrainDF)
//        val stemmedTrainDF = stemmer.setInputCol("search_filtered").setOutputCol("search_stemmed")
//          .transform(semiStemmedTrainDF)
//
//        // TF-IDF training data
//        val semiFeaturesTrainDF = hashingTF.setInputCol("title_stemmed").setOutputCol("title_features")
//          .setNumFeatures(10000).transform(stemmedTrainDF)
//        val featuresTrainDF = hashingTF.setInputCol("search_stemmed").setOutputCol("search_features")
//          .setNumFeatures(10000).transform(semiFeaturesTrainDF)
//        val semiRescaledTRAINDF = idf.setInputCol("title_features").setOutputCol("title_tfidf").fit(featuresTrainDF)
//          .transform(featuresTrainDF)
//        val rescaledTrainDF = idf.setInputCol("search_features").setOutputCol("search_tfidf").fit(semiRescaledTRAINDF)
//          .transform(semiRescaledTRAINDF)
//
//        // Word2Vec to training data
//        val semiVecTrainDF = word2Vec.setInputCol("title_stemmed").setOutputCol("title_w2v").fit(rescaledTrainDF)
//          .transform(rescaledTrainDF)
//        val vecTrainDF = word2Vec.setInputCol("search_stemmed").setOutputCol("search_w2v").fit(semiVecTrainDF)
//          .transform(semiVecTrainDF)
//        vecTrainDF.printSchema()
////        vecTrainDF.take(10).foreach(x => println(x))
//
//        //      -----------------------------Description Data Preprocess-----------------------------------------------
//        // Tokenizing the description data
//        val tokenizedDescriptionDF = tokenizer.setInputCol("product_description").setOutputCol("description_tokenized")
//          .transform(descDF)
//
//        // Removing stopwords from description data
//        val filteredDescriptionDF = remover.setInputCol("description_tokenized").setOutputCol("description_filtered")
//          .transform(tokenizedDescriptionDF)
//
//        // Stemming description data
//        val stemmedDescriptionDF = stemmer.setInputCol("description_filtered").setOutputCol("description_stemmed")
//          .transform(filteredDescriptionDF)
//
//        // TF-IDF description data
//        val featuresDescriptionDF = hashingTF.setInputCol("description_stemmed").setOutputCol("description_features").setNumFeatures(10000)
//          .transform(stemmedDescriptionDF)
//        val rescaledDescriptionDF = idf.setInputCol("description_features").setOutputCol("description_tfidf").fit(featuresDescriptionDF)
//          .transform(featuresDescriptionDF)
//
//        // Word2Vec to description data
//        val vecDescDF = word2Vec.setInputCol("description_stemmed").setOutputCol("description_w2v")
//          .fit(rescaledDescriptionDF).transform(rescaledDescriptionDF)//.withColumnRenamed("product_uid", "uid")
//        vecDescDF.printSchema()
//        vecDescDF.take(10).foreach(x => println(x))
//
//
//        //-----------------------------Attribute Data Preprocess-----------------------------------------------
//
//        val result1 = attrDF.groupBy("product_uid").agg(collect_set("name"))
//        val result2 = attrDF.groupBy("product_uid").agg(collect_set("value"))
//        val result3 = result1.join(result2, result2("product_uid") === result1("product_uid"), "inner")
//          .select(result1("product_uid"), result1("collect_set(name)"), result2("collect_set(value)"))
//        val result4 = result3
//          .withColumn("name", concat_ws(" ", result3("collect_set(name)")))
//          .withColumn("value", concat_ws(" ", result3("collect_set(value)")))
//          .orderBy("product_uid").select("product_uid", "name", "value")
//        val result5 = result4
//          .withColumn("attributes", concat(col("name"), lit(" "), col("value")))
//          .select(col("product_uid"), col("attributes"))
//
//        // Tokenizing the attribute data
//        val tokenizedAttributeDF = tokenizer.setInputCol("attributes").setOutputCol("attr_tokenized")
//          .transform(result5)
//
//        // Removing stopwords from attribute data
//        val filteredAttributeDF = remover.setInputCol("attr_tokenized").setOutputCol("attr_filtered")
//          .transform(tokenizedAttributeDF)
//
//        // Stemming description data
//        val stemmedAttributeDF = stemmer.setInputCol("attr_filtered").setOutputCol("attr_stemmed")
//          .transform(filteredAttributeDF).withColumnRenamed("product_uid", "uid")
//
////        // TF-IDF attribute data
////        val featuresAttributeDF = hashingTF.setInputCol("attr_stemmed").setOutputCol("attr_features").setNumFeatures(10000)
////          .transform(stemmedAttributeDF)
////        val rescaledAttributeDF = idf.setInputCol("attr_features").setOutputCol("attr_tfidf").fit(featuresAttributeDF)
////          .transform(featuresAttributeDF)
////
////        // Word2Vec to attribute data
////        val vecAttrDF = word2Vec.setInputCol("attr_stemmed").setOutputCol("attr_w2v").fit(rescaledAttributeDF)
////          .transform(rescaledAttributeDF).withColumnRenamed("product_uid", "uid")
////        vecAttrDF.printSchema()
////        vecAttrDF.take(10).foreach(x => println(x))
//
//        //-----------------------------------------Joining the data-----------------------------------------
//        val joinedTrainDescDef = vecTrainDF.join(vecDescDF, usingColumn = "product_uid")
////        joinedTrainDescDef.printSchema()
//        val joinAllDF = joinedTrainDescDef
//          .join(stemmedAttributeDF,joinedTrainDescDef("product_uid")===stemmedAttributeDF("uid"), "left_outer")
//          .drop("uid").withColumn("attr_filled", replaceNull(col("attr_stemmed")))
////        joinAllDF.printSchema()
////        joinAllDF.show(false)
//
//        val featuresAllDF = hashingTF.setInputCol("attr_filled").setOutputCol("attr_features").setNumFeatures(10000)
//          .transform(joinAllDF)
//        val rescaledAllDF = idf.setInputCol("attr_features").setOutputCol("attr_tfidf").fit(featuresAllDF)
//          .transform(featuresAllDF)
//        val vecAllDF = word2Vec.setInputCol("attr_filled").setOutputCol("attr_w2v").fit(rescaledAllDF)
//          .transform(rescaledAllDF)
//
//        val finalDF = vecAllDF
//          .withColumn("cos_title_t", cosineSimilarity(col("title_tfidf"), col("search_tfidf")))
//          .withColumn("cos_desc_t", cosineSimilarity(col("description_tfidf"), col("search_tfidf")))
//          .withColumn("cos_attr_t", cosineSimilarity(col("attr_tfidf"), col("search_tfidf")))
//          .withColumn("cos_title_w", cosineSimilarity(toSparse(col("title_w2v")), toSparse(col("search_w2v"))))
//          .withColumn("cos_desc_w", cosineSimilarity(toSparse(col("description_w2v")), toSparse(col("search_w2v"))))
//          .withColumn("cos_attr_w", cosineSimilarity(toSparse(col("attr_w2v")), toSparse(col("search_w2v"))))
//          .withColumn("jacc_title", jaccardSimilarity(col("title_stemmed"),col("search_stemmed")))
//          .withColumn("jacc_desc", jaccardSimilarity(col("description_stemmed"),col("search_stemmed")))
//          .withColumn("jacc_attr", jaccardSimilarity(col("attr_filled"),col("search_stemmed")))
//          .withColumn("comm_title", commonWords(col("title_stemmed"),col("search_stemmed")))
//          .withColumn("comm_desc", commonWords(col("description_stemmed"),col("search_stemmed")))
//          .withColumn("comm_attr", commonWords(col("attr_filled"),col("search_stemmed")))
//          .withColumn("query_length", queryLength(col("search_stemmed")))
//          .withColumn("title_ratio", ratio(col("comm_title"), col("query_length")))
//          .withColumn("desc_ratio", ratio(col("comm_desc"), col("query_length")))
//          .withColumn("attr_ratio", ratio(col("comm_attr"), col("query_length")))
//          .withColumn("label", udfToDouble(col("relevance")))
//          .drop("relevance")
//          .select(col("product_uid"), col("cos_title_t"), col("cos_desc_t"),
//              col("cos_attr_t"), col("cos_title_w"), col("cos_desc_w"),
//              col("cos_attr_w"), col("jacc_title"), col("jacc_desc"),
//              col("jacc_attr"), col("comm_title"), col("comm_desc"),
//              col("comm_attr"), col("query_length"), col("title_ratio"),
//              col("desc_ratio"), col("attr_ratio"), col("label"),
//              col("title_tfidf"), col("search_tfidf"), col("title_w2v"),
//              col("search_w2v")
//          )
//
//        finalDF.write.parquet("finaldf.parquet")

        val finalDF = spark.read.parquet("finaldf.parquet")

        println(finalDF.count())

        //-----------------------------------------Test Area 51-----------------------------------------
        for(i <- 1 to 25) {//2, 3, 5, 6, 7, 10, 13, 16, 17, 20, 21, 22
//            println(i)
//            val assembler = new VectorAssembler()
//              .setInputCols(MyUtilities.testArray(i))
//              .setOutputCol("features")
//            // Transform the DataFrame
//            val output = assembler.transform(finalDF)
//            println(output.count())
//
//            // Splitting the data by create an array of the training and test data
//            val Array(training, test) = output.select("label", "features").
//              randomSplit(Array(0.6, 0.4), seed = 12345) //, seed = 1L
//            println(training.count())
//            println(test.count())

            // Linear Regression model
//            val lr = new LinearRegression()
//              .setFeaturesCol("features")
//              .setLabelCol("label")
//              .setSolver("l-bfgs")
//              .setRegParam(0.1)
//
//
//            val lirModel = lr.fit(training)
//            val result = lirModel.transform(test)

            // Decision Tree model
//            val dt = new DecisionTreeRegressor()
//              .setFeaturesCol("features")
//              .setLabelCol("label")
//              .setMaxDepth(5)
//
//            val dtModel = dt.fit(training)
//            val result = dtModel.transform(test)

            //         Random Forrest model
//            val rf = new RandomForestRegressor()
//              .setFeaturesCol("features")
//              .setLabelCol("label")
//              .setMaxDepth(10)
//
//            val rfModel = rf.fit(training)
//            val result = rfModel.transform(test)

            // Gradient Boost model
//            val gbt = new GBTRegressor()
//              .setFeaturesCol("features")
//              .setLabelCol("label")
////              .setMaxIter(100)
//
//            val gbtModel = gbt.fit(training)
//            val result = gbtModel.transform(test)

            //Creation of the evaluator
//            val evaluator = new RegressionEvaluator()
//              .setLabelCol("label")
//              .setPredictionCol("prediction")
//              .setMetricName("mse")
//
//            println(result.count())
//
//            val MSElr = evaluator.evaluate(result)
//            println("Linear Regression MSE = " + MSElr)
        }

        //-------------------------------- Test Area 52 ---------------------------------------------
        val final2DF = finalDF.withColumn("relevance", toDual(col("label")).cast(DoubleType))
//        final2DF.write.csv("outfile.csv")
//        final2DF.select("label", "relevance", "cos_title_t", "cos_desc_t", "cos_attr_t"
//            ,"cos_title_w", "cos_desc_w", "cos_attr_w"
//            ,"jacc_title", "jacc_desc", "jacc_attr").write.csv("outfile.csv")

        for (i <- 0 to 25) {
            println(i)
            val assembler = new VectorAssembler()
              .setInputCols(MyUtilities.testArray(i))
              .setOutputCol("features")

            val dataset = assembler.transform(final2DF)
            println(dataset.count())

//             K-Means model
            val km = new KMeans().setFeaturesCol("features").setK(2) //.setSeed(1L)
              //.setMaxIter(20)
            val model = km.fit(dataset)
            model.clusterCenters.foreach(println)

            val predictions = model.transform(dataset)

            // BKM model
//            val bkm = new BisectingKMeans().setK(2).setSeed(1)
//            val model = bkm.fit(dataset)
//            val predictions = model.transform(dataset)

    //        predictions.select("prediction", "relevance").show(false)
    //        println("Cluster Centers: ")
    //        model.clusterCenters.foreach(println)
    //
    //        val evaluator = new ClusteringEvaluator()
    //        val silhouette = evaluator.evaluate(predictions)
    //
    //        println(s"Silhouette with squared euclidean distance = $silhouette")

            // Evaluation

//            val predictions_double = predictions.withColumn("prediction_d", col("prediction").cast(DoubleType))
//
//            val TN: Double = predictions.select("relevance", "prediction").filter("relevance = 0 and prediction = 0").count
//            val TP: Double = predictions.select("relevance", "prediction").filter("relevance = 1 and prediction = 1").count
//            val FN: Double = predictions.select("relevance", "prediction").filter("relevance = 1 and prediction = 0").count
//            val FP: Double = predictions.select("relevance", "prediction").filter("relevance = 0 and prediction = 1").count
//            val total = predictions.select("relevance").count.toDouble
//
//            val confusion: Matrix = Matrices.dense(2, 2, Array(TP, FN, FP, TN))
//
//            val accuracy = (TP + TN) / total
//            val precision = TP / (TP + FP)
//            val recall = TP / (TP + FN)
//            val F1 = 2 / (1 / precision + 1 / recall)
//
//            println(confusion)
//            println("Accuracy: ", accuracy)
//            println("Precision: ", precision)
//            println("Recall: ", recall)
//            println("F1: ", F1)
//
//            val evaluator = new MulticlassClassificationEvaluator()
//              .setLabelCol("relevance")
//              .setPredictionCol("prediction_d")
//
//            println("F1 of evaluator: ", evaluator.evaluate(predictions_double))

        }

        //----------------------------------- Thresholds ----------------------------------------------

        val customPrediction = udf[Int, Double](MyUtilities.customPrediction)


        val final4DF = final2DF.withColumn("prediction", customPrediction(col("title_ratio")))

        val TN = final4DF.select("relevance", "prediction").filter("relevance = 0 and prediction = 0").count.toDouble
        val TP = final4DF.select("relevance", "prediction").filter("relevance = 1 and prediction = 1").count.toDouble
        val FN = final4DF.select("relevance", "prediction").filter("relevance = 1 and prediction = 0").count.toDouble
        val FP = final4DF.select("relevance", "prediction").filter("relevance = 0 and prediction = 1").count.toDouble
        val total = final4DF.select("relevance").count.toDouble

        val confusion: Matrix = Matrices.dense(2, 2, Array(TP, FN, FP, TN))

        val accuracy = (TP + TN) / total
        val precision = TP / (TP + FP)
        val recall = TP / (TP + FN)
        val F1 = 2 / (1 / precision + 1 / recall)

        println(confusion)
        println("Accuracy: ", accuracy)
        println("Precision: ", precision)
        println("Recall: ", recall)
        println("F1: ", F1)

    }

}
